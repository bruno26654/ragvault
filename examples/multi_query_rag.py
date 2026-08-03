#!/usr/bin/env python3
"""Multi-query RAG: composed questions, version precedence, safe citations.

Runs fully offline by default. To use a real LLM for decomposition and the
answer, export GROQ_API_KEY and install the (optional) client:

    pip install groq
    export GROQ_API_KEY=...
    python examples/multi_query_rag.py --groq

RagVault never imports or requires any provider: `llm` and `decompose` are
plain Python callables you supply.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import tempfile

import ragvault

CORPUS = [
    # Current policy plus the revoked version it replaced.
    {"id": "refund-2024", "text":
     "Refund requests must be filed within 30 days of purchase. Approved "
     "refunds are returned to the original payment method.",
     "metadata": {"status": "VIGENTE", "doc_group": "refund",
                  "effective_date": "2024-01-01", "version": 2,
                  "doc_type": "policy"}},
    {"id": "refund-2019", "text":
     "Refund requests must be filed within 90 days of purchase. Approved "
     "refunds are issued as store credit.",
     "metadata": {"status": "REVOGADO", "doc_group": "refund",
                  "effective_date": "2019-01-01", "version": 1,
                  "doc_type": "policy"}},
    # Evidence for the second facet of the question.
    {"id": "processing-time", "text":
     "Approved reimbursements are processed within five business days and "
     "appear on the statement in the next billing cycle.",
     "metadata": {"status": "VIGENTE", "doc_type": "policy"}},
    # Semantically similar distractors.
    {"id": "tax-refunds", "text":
     "Government tax refunds follow a separate federal schedule and are not "
     "handled by our support team.",
     "metadata": {"status": "VIGENTE", "doc_type": "faq"}},
    {"id": "gift-cards", "text":
     "Gift cards are non-refundable and cannot be cancelled once issued.",
     "metadata": {"status": "VIGENTE", "doc_type": "faq"}},
]

QUESTION = "How long do I have to request a refund, and how long until I get the money?"


def offline_decomposer(question: str) -> list[str]:
    """Rule-based splitter so the example runs with no network. A real LLM
    decomposer has the same signature: question -> list[str]."""
    parts = [p.strip(" ?.") for p in re.split(r",| and ", question)]
    return [p for p in parts if len(p.split()) >= 3]


def offline_llm(prompt: str) -> str:
    """Deterministic stand-in for an LLM.

    It answers *from the context it was given*, citing the block that actually
    contains each fact — never a hardcoded number. That is exactly what a real
    LLM must do, and it keeps this example honest: if retrieval fails to bring
    a fact in, the stand-in says so instead of inventing a citation.
    """
    blocks = re.findall(r"^\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[|\n\n#|\Z)",
                        prompt, re.DOTALL | re.MULTILINE)

    def cite(needle: str) -> str | None:
        for index, body in blocks:
            if needle in body:
                return index
        return None

    deadline, payout = cite("30 days"), cite("five business days")
    parts = []
    if deadline:
        parts.append(f"You have 30 days to request a refund [{deadline}]")
    if payout:
        parts.append(
            f"approved reimbursements are processed within five business "
            f"days [{payout}]"
        )
    if not parts:
        return "The context does not contain the answer."
    return ", and ".join(parts) + "."


def offline_verifier(payload):
    """Deterministic stand-in for an LLM judge.

    Checks each claim's numbers against the block it cites — enough to show
    the contract without a network call. A real judge (see --groq) reasons
    semantically instead.

    Facet coverage is deliberately *not* faked: deciding whether "how long
    until I get the money" was answered by "processed within five business
    days" is a semantic judgement with no lexical overlap to lean on. The
    stand-in says it cannot judge, so `complete` comes back False — an
    unevaluated facet is never a covered one.
    """
    out = []
    for item in payload["claims"]:
        claim = item["claim"]
        if not item["citations"]:
            out.append({"verdict": "uncited", "rationale": "no [n] marker"})
            continue
        # The cited blocks come with the claim; no need to re-parse the
        # context to find them, and no way to drift onto an uncited block.
        cited = " ".join(e["text"] for e in item["evidence"])
        numbers = set(re.findall(r"\d+", claim))
        if numbers and not numbers & set(re.findall(r"\d+", cited)):
            out.append({
                "verdict": "contradicted",
                "rationale": (f"claim states {sorted(numbers)} but the cited "
                              "block does not contain those figures"),
            })
        else:
            out.append({"verdict": "supported",
                        "rationale": "figures match the cited block"})
    return {
        "claims": out,
        "facets": [{"facet": facet, "covered": False,
                    "rationale": "the offline stand-in cannot judge coverage "
                                 "semantically; run with --groq"}
                   for facet in payload["facets"]],
    }


def groq_callables():
    """Real Groq-backed decomposer + answerer (optional dependency)."""
    from groq import Groq  # imported only when --groq is passed

    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    model = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

    def complete(prompt: str, *, temperature: float = 0.0) -> str:
        response = client.chat.completions.create(
            model=model,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.choices[0].message.content or ""

    def decompose(question: str) -> list[str]:
        prompt = (
            "Split the question into the minimal set of independent search "
            "queries needed to answer it fully. Each query must cover exactly "
            "one thing the answer owes — never combine two into one, since a "
            "composite query spends a single context slot on two pieces of "
            "evidence and usually returns only one. Reply with a JSON array "
            f"of strings and nothing else.\n\nQuestion: {question}"
        )
        raw = complete(prompt).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        # A malformed reply raises here; retrieve_multi catches it and falls
        # back to the single original question.
        return [str(s) for s in json.loads(match.group(0) if match else raw)]

    def verify(payload):
        """LLM-as-judge: it segments and classifies, and never rewrites.

        A verifier that proposes a correction and then approves it is grading
        its own text, so `replacement` stays out of the contract entirely —
        a claim that does not hold is removed, not rephrased.
        """
        # Each claim is shown with the sources *it* cited, so support cannot be
        # borrowed from a block the claim never named. The full context still
        # goes in: judging `uncited`, or noticing that the right evidence was
        # available under another marker, needs it.
        cited = "\n\n".join(
            f"## Claim {i}: {c['claim']}\n" + (
                "\n".join(f"[{e['index']}] ({e['metadata']}) {e['text']}"
                          for e in c["evidence"])
                or "(this claim cites no source)"
            )
            for i, c in enumerate(payload["claims"], start=1)
        )
        prompt = (
            "You verify a RAG answer. Do not rewrite anything: segment and "
            "classify only.\n\n"
            f"# Full context\n{payload['context']}\n\n"
            f"# Sources each claim actually cited\n{cited}\n\n"
            f"# Question asked by the user\n{payload['question']}\n\n"
            f"# Answer\n{payload['answer']}\n\n"
            f"# Facets the answer was supposed to cover\n{payload['facets']}\n\n"
            "1. Segment the answer into EVERY material proposition: verbatim "
            "spans of the answer, in order, non-overlapping, each carrying a "
            "single proposition (one sentence stating two facts becomes two "
            "items).\n"
            "2. Classify each with 'verdict' (one of "
            f"{payload['verdicts']}) and 'rationale', judging against the "
            "explicit facts of the question, the cited blocks and the source "
            "metadata:\n"
            # The verdict semantics live in the prompt, not in the library:
            # only a model can judge these, and hard-coding rules for them
            # would be domain guesswork.
            "   - 'contradicted' also applies when the claim contradicts a "
            "fact the user stated in the question, even with no document "
            "involved.\n"
            "   - 'question_fact' is ONLY for restating something the "
            "question asserted; a fact the question supplied is never missing "
            "or undeterminable. Conclusions, rule applications and deductions "
            "are 'inference'.\n"
            "   - A claim about a past, superseded or historical rule is only "
            "'supported' when some cited block's metadata actually shows that "
            "older state. Differing from the current rule does not prove an "
            "older rule existed.\n"
            "3. Every verdict must name its ground, and RagVault checks it: "
            "add 'quote' with the exact words that carry it, copied verbatim "
            "— from the cited source for 'supported', from the question for "
            "'question_fact', from either for 'inference' and "
            "'contradicted'. An invented quote fails the claim.\n"
            "4. For each facet, 'covered' is true only when ALL of its "
            "components were answered correctly.\n"
            "Omitting a proposition or a facet is not a pass. Reply with JSON "
            'only: {"claims": [{"claim", "verdict", "rationale", "quote"}], '
            '"facets": [{"facet", "covered", "rationale"}]}.'
        )
        raw = complete(prompt).strip()
        match = re.search(r"[\[{].*[\]}]", raw, re.DOTALL)
        return json.loads(match.group(0) if match else raw)

    return decompose, complete, verify


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groq", action="store_true",
                        help="use Groq for decomposition and the answer")
    args = parser.parse_args()

    if args.groq:
        decompose, llm, verify = groq_callables()
    else:
        decompose, llm, verify = offline_decomposer, offline_llm, offline_verifier

    with tempfile.TemporaryDirectory() as tmp:
        with ragvault.open(tmp, preset="offline-lite") as kb:
            kb.add(CORPUS)

            answer = kb.ask_multi(
                QUESTION,
                llm=llm,
                decompose=decompose,
                max_subqueries=6,
                fusion="weighted_rrf",
                filters={"status": "VIGENTE"},   # revoked docs never searched
                resolve_versions=True,           # precedence among versions
                citations=True,
                verify=verify,                   # post-generation validation
                verification_mode="repair",
                explain=True,
                trace=True,
                k=5,
            )

            result = answer.result
            print("Question:", QUESTION)
            print("\nSubqueries actually run:")
            for q in result.subqueries:
                print("  -", q)

            print("\nAnswer:\n ", answer.text)

            print("\nCitations (real stored chunks only):")
            for c in result.citations:
                print(f"  [{c.index}] {c.document_id} v{c.document_version} "
                      f"chunks={c.chunk_ids}")

            if result.conflicts:
                print("\nVersion conflicts resolved:")
                for conflict in result.conflicts:
                    kept = conflict["kept"]["document_id"]
                    for dropped in conflict["dropped"]:
                        print(f"  {conflict['group']}: kept {kept}, dropped "
                              f"{dropped['document_id']} — {dropped['reason']}")

            # Fidelity (ok) and completeness (complete) are separate axes:
            # every claim can be supported and a facet still go unanswered.
            report = answer.verification
            print(f"\nVerification ({report.mode}): ok={report.ok} "
                  f"complete={report.complete} {report.counts()} in "
                  f"{report.elapsed_ms:.1f} ms")
            for claim in report.claims:
                print(f"  [{claim.verdict}/{claim.action}] {claim.claim}")
                if claim.rationale:
                    print(f"      ↳ {claim.rationale}")
            for facet in report.uncovered_facets:
                print(f"  [uncovered] {facet['facet']}")
                if facet["rationale"]:
                    print(f"      ↳ {facet['rationale']}")

            print("\nStage timings (ms):", result.trace["stage_ms"])
            print("Documents in context:", result.documents)
            assert "refund-2019" not in result.documents, "revoked must not be cited"


if __name__ == "__main__":
    main()
