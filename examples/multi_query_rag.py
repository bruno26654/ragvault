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
            "queries needed to answer it fully. Reply with a JSON array of "
            f"strings and nothing else.\n\nQuestion: {question}"
        )
        raw = complete(prompt).strip()
        match = re.search(r"\[.*\]", raw, re.DOTALL)
        # A malformed reply raises here; retrieve_multi catches it and falls
        # back to the single original question.
        return [str(s) for s in json.loads(match.group(0) if match else raw)]

    return decompose, complete


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groq", action="store_true",
                        help="use Groq for decomposition and the answer")
    args = parser.parse_args()

    if args.groq:
        decompose, llm = groq_callables()
    else:
        decompose, llm = offline_decomposer, offline_llm

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

            print("\nStage timings (ms):", result.trace["stage_ms"])
            print("Documents in context:", result.documents)
            assert "refund-2019" not in result.documents, "revoked must not be cited"


if __name__ == "__main__":
    main()
