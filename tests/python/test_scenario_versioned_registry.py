"""A demanding end-to-end scenario: large, noisy, versioned corpus.

Everything the pipeline claims to do at once, on a corpus built to punish each
of them:

* **versioned** — current rules with revoked predecessors that state *different
  numbers*, so citing the wrong version is visible in the answer;
* **redundant** — near-duplicates of the current rule under different ids;
* **incomplete** — documents that name a topic without stating its value, so
  the only correct answer is "not determinable from the sources";
* **partially conflicting** — two documents that are equally current and say
  different things, where metadata cannot decide which wins;
* **noisy** — hundreds of unrelated documents plus distractors that mention the
  entities in irrelevant contexts.

Questions are multi-faceted and span several entities, deadlines, procedures
and conditions that must not be mixed. The stand-in model and judge are
deterministic and derive everything from the context they are given: nothing
here asserts a fact the pipeline did not actually retrieve.
"""

from __future__ import annotations

import re

import pytest

import ragvault

# --- the corpus -----------------------------------------------------------
#
# Four entities, four topics. Distinct tokens so a wrong entity in an answer is
# unambiguous rather than a matter of interpretation.

ENTITIES = ("ALPHA", "BETA", "GAMMA", "DELTA")

#: (doc_id, text, metadata) for the documents the questions are about.
CORE_DOCS = [
    # ALPHA registration: current rule plus two revoked predecessors, each with
    # a different number. Citing a revoked one shows up as a wrong figure.
    ("alpha-reg-v3",
     "Program ALPHA registration. The registration deadline for ALPHA is 45 "
     "calendar days from the notice of eligibility. Late registrations are "
     "rejected without review.",
     {"entity": "ALPHA", "topic": "registration", "doc_group": "alpha-reg",
      "status": "VIGENTE", "effective_date": "2025-01-01", "version": 3}),
    ("alpha-reg-v2",
     "Program ALPHA registration. The registration deadline for ALPHA is 60 "
     "calendar days from the notice of eligibility.",
     {"entity": "ALPHA", "topic": "registration", "doc_group": "alpha-reg",
      "status": "REVOGADO", "effective_date": "2023-01-01", "version": 2}),
    ("alpha-reg-v1",
     "Program ALPHA registration. The registration deadline for ALPHA is 90 "
     "calendar days from the notice of eligibility.",
     {"entity": "ALPHA", "topic": "registration", "doc_group": "alpha-reg",
      "status": "REVOGADO", "effective_date": "2021-01-01", "version": 1}),
    # Redundant restatement of the current ALPHA rule under another id.
    ("alpha-reg-handbook",
     "Program ALPHA registration. The registration deadline for ALPHA is 45 "
     "calendar days from the notice of eligibility. See the handbook for "
     "worked examples.",
     {"entity": "ALPHA", "topic": "registration", "doc_group": "alpha-reg",
      "status": "VIGENTE", "effective_date": "2025-01-01", "version": 3}),

    # BETA appeal: current rule plus a revoked one with a different figure.
    ("beta-appeal-v2",
     "Program BETA appeals. An appeal against a BETA decision must be filed "
     "within 15 business days of notification, addressed to the review board.",
     {"entity": "BETA", "topic": "appeal", "doc_group": "beta-appeal",
      "status": "VIGENTE", "effective_date": "2024-06-01", "version": 2}),
    ("beta-appeal-v1",
     "Program BETA appeals. An appeal against a BETA decision must be filed "
     "within 30 business days of notification.",
     {"entity": "BETA", "topic": "appeal", "doc_group": "beta-appeal",
      "status": "REVOGADO", "effective_date": "2020-06-01", "version": 1}),

    # GAMMA fee: stated. GAMMA appeal procedure: named but never stated, so the
    # only correct answer about it is that the sources do not determine it.
    ("gamma-fee-v1",
     "Program GAMMA charges. The GAMMA application fee is 250 credits, "
     "payable at submission.",
     {"entity": "GAMMA", "topic": "fee", "doc_group": "gamma-fee",
      "status": "VIGENTE", "effective_date": "2024-02-01", "version": 1}),
    ("gamma-appeal-stub",
     "Program GAMMA appeals. The GAMMA appeal procedure is described in "
     "Annex IV of the operating manual.",
     {"entity": "GAMMA", "topic": "appeal", "doc_group": "gamma-appeal",
      "status": "VIGENTE", "effective_date": "2024-02-01", "version": 1}),

    # DELTA eligibility: two documents, equally current, equally versioned,
    # same effective date — and they disagree. Metadata cannot decide.
    ("delta-elig-a",
     "Program DELTA eligibility. An applicant qualifies for DELTA after 24 "
     "months of continuous participation.",
     {"entity": "DELTA", "topic": "eligibility", "doc_group": "delta-elig",
      "status": "VIGENTE", "effective_date": "2025-03-01", "version": 2}),
    ("delta-elig-b",
     "Program DELTA eligibility. An applicant qualifies for DELTA after 36 "
     "months of continuous participation.",
     {"entity": "DELTA", "topic": "eligibility", "doc_group": "delta-elig",
      "status": "VIGENTE", "effective_date": "2025-03-01", "version": 2}),
]

#: Documents that mention the entities in contexts that answer nothing —
#: lexically close, semantically useless.
DISTRACTORS = [
    ("alpha-newsletter",
     "Program ALPHA newsletter. The ALPHA community call happens every 45 "
     "days; registration for the call is open to all members.",
     {"entity": "ALPHA", "topic": "newsletter", "status": "VIGENTE"}),
    ("beta-travel",
     "Program BETA travel policy. BETA reimbursement requests are filed "
     "within 15 business days of travel, using the expenses portal.",
     {"entity": "BETA", "topic": "travel", "status": "VIGENTE"}),
    ("gamma-fee-history",
     "Program GAMMA charges. Historical note: the GAMMA fee was reviewed in "
     "2019 and again in 2022 by the finance committee.",
     {"entity": "GAMMA", "topic": "fee-history", "status": "VIGENTE"}),
    ("delta-glossary",
     "Program DELTA glossary. 'Continuous participation' means participation "
     "without a gap longer than 60 days.",
     {"entity": "DELTA", "topic": "glossary", "status": "VIGENTE"}),
    ("cross-summary",
     "Programs overview. ALPHA, BETA, GAMMA and DELTA are administered by the "
     "same secretariat and share a single appeals calendar.",
     {"entity": "ALL", "topic": "overview", "status": "VIGENTE"}),
]

TOPIC_WORDS = {
    "registration": ("registration deadline",),
    "appeal": ("appeal",),
    "fee": ("fee",),
    "eligibility": ("qualifies", "eligibility"),
}


def _noise(count: int) -> list[dict]:
    """Unrelated documents, worded like the real ones so they compete."""
    subjects = ("procurement", "onboarding", "archival", "training",
                "accessibility", "data retention", "vendor review")
    out = []
    for i in range(count):
        subject = subjects[i % len(subjects)]
        out.append({
            "id": f"noise-{i:04d}",
            "text": (f"Secretariat circular {i}. The {subject} deadline is "
                     f"{10 + i % 80} calendar days from notification. This "
                     f"circular applies to all secretariat procedures and "
                     f"supersedes circular {max(0, i - 40)}."),
            "metadata": {"entity": "SECRETARIAT", "topic": subject,
                         "status": "VIGENTE", "effective_date": "2024-01-01"},
        })
    return out


@pytest.fixture(scope="module")
def registry(tmp_path_factory):
    path = tmp_path_factory.mktemp("registry") / "kb"
    kb = ragvault.open(path)
    docs = [{"id": i, "text": t, "metadata": m}
            for i, t, m in CORE_DOCS + DISTRACTORS]
    kb.add(docs + _noise(900))
    kb.flush()
    yield kb
    kb.close()


# --- deterministic stand-ins ----------------------------------------------


def _blocks(context: str) -> list[tuple[int, str]]:
    """(index, body) for each numbered context block."""
    return [(int(n), body) for n, body in re.findall(
        r"^\[(\d+)\][^\n]*\n(.*?)(?=\n\n\[|\n\n#|\Z)",
        context, re.DOTALL | re.MULTILINE)]


def _sentence_for(body: str, entity: str, topic: str) -> str | None:
    """The sentence of a block that states `topic` for `entity`, if any."""
    for sentence in re.split(r"(?<=\.)\s+", body):
        if entity not in sentence:
            continue
        if any(word in sentence.lower() for word in TOPIC_WORDS[topic]):
            if re.search(r"\d", sentence):
                return sentence.strip()
    return None


def answerer(facets: list[tuple[str, str]]):
    """A model that answers strictly from the context it is handed.

    One sentence per facet, each citing the block the sentence came from. A
    facet the context does not state is declared undetermined rather than
    guessed — the behaviour a correct RAG answer must have, reproduced without
    a network call.
    """
    def llm(prompt: str) -> str:
        context = prompt.split("# Context\n", 1)[-1]
        parts = []
        for entity, topic in facets:
            found = None
            for index, body in _blocks(context):
                sentence = _sentence_for(body, entity, topic)
                if sentence:
                    found = (index, sentence)
                    break
            if found:
                parts.append(f"{found[1]} [{found[0]}]")
            else:
                parts.append(
                    f"The sources do not determine the {topic} for {entity}."
                )
        return " ".join(parts)
    return llm


def judge(payload: dict) -> dict:
    """A judge that only checks what is checkable from the evidence it got.

    Three structural tests, no domain knowledge: the claim's entity must match
    the cited source's `entity` metadata, its figures must appear in the cited
    text, and a claim citing nothing is `uncited` — except a declared
    non-answer, which asserts nothing about a source.
    """
    claims = []
    for item in payload["claims"]:
        # The citation marker is provenance, not content: counting `[3]` as a
        # figure of the claim would ask the source to contain the number 3.
        claim = re.sub(r"\[\d+\]", "", item["claim"])
        cited_text = " ".join(e["text"] for e in item["evidence"])
        cited_entities = {e["metadata"].get("entity") for e in item["evidence"]}

        if "do not determine" in claim:
            claims.append({"verdict": "supported",
                           "rationale": "declared non-answer, asserts nothing"})
            continue
        if not item["evidence"]:
            claims.append({"verdict": "uncited", "rationale": "no source"})
            continue

        named = [e for e in ENTITIES if e in claim]
        if named and not (set(named) & cited_entities):
            claims.append({
                "verdict": "contradicted",
                "rationale": (f"claim is about {named} but cites sources about "
                              f"{sorted(x for x in cited_entities if x)}"),
            })
            continue

        figures = set(re.findall(r"\d+", claim))
        if figures and not figures <= set(re.findall(r"\d+", cited_text)):
            claims.append({"verdict": "unsupported",
                           "rationale": "figures not in the cited source"})
            continue

        quote = next((s for s in re.split(r"(?<=\.)\s+", cited_text)
                      if figures and figures & set(re.findall(r"\d+", s))), None)
        claims.append({"verdict": "supported", "rationale": "matches source",
                       **({"quote": quote} if quote else {})})

    facets = [
        {"facet": facet,
         "covered": all(
             token in payload["answer"]
             for token in (facet.split()[0],)
         ) and "do not determine" not in payload["answer"],
         "rationale": "entity appears in an answered claim"}
        for facet in payload["facets"]
    ]
    return {"claims": claims, "facets": facets}


# --- retrieval and version resolution -------------------------------------


class TestRetrievalUnderNoise:
    def test_current_rules_are_found_for_two_entities_at_once(self, registry):
        result = registry.retrieve_multi(
            "registration deadline for ALPHA and appeal deadline for BETA",
            subqueries=["ALPHA registration deadline",
                        "BETA appeal deadline"],
            resolve_versions=True, k=8,
        )
        assert "alpha-reg-v3" in result.documents
        assert "beta-appeal-v2" in result.documents

    def test_revoked_versions_never_reach_the_context(self, registry):
        result = registry.retrieve_multi(
            "ALPHA registration deadline",
            subqueries=["ALPHA registration deadline"],
            resolve_versions=True, k=8,
        )
        assert "alpha-reg-v2" not in result.documents
        assert "alpha-reg-v1" not in result.documents
        assert "60 calendar days" not in result.context
        assert "90 calendar days" not in result.context

    def test_every_dropped_document_is_named_with_a_reason(self, registry):
        result = registry.retrieve_multi(
            "ALPHA registration deadline",
            subqueries=["ALPHA registration deadline"],
            resolve_versions=True, k=8, explain=True,
        )
        conflict = next(c for c in result.conflicts if c["group"] == "alpha-reg")
        dropped = {d["document_id"]: d["reason"] for d in conflict["dropped"]}
        assert {"alpha-reg-v1", "alpha-reg-v2"} <= set(dropped)
        assert all(reason for reason in dropped.values())


class TestRevocationIsAbsolute:
    """A document whose status says it is superseded is not the rule in force,
    whether or not its replacement happened to be retrieved. Judged only
    *relatively*, a revoked rule that outranked its own successor reached the
    context looking current, with nothing reported."""

    def test_a_revoked_rule_alone_in_its_group_is_still_dropped(self, registry):
        result = registry.retrieve_multi(
            "BETA appeal deadline",
            # Two subqueries so BETA's slice of the pool is small enough that
            # the revoked document can outrank its own successor.
            subqueries=["BETA appeal notification", "secretariat deadline"],
            resolve_versions=True, k=8, explain=True,
        )
        assert "beta-appeal-v1" not in result.documents
        assert "30 business days" not in result.context

    def test_the_replacement_takes_the_freed_slot(self, registry):
        """Eliminating a candidate must not shrink the context: the successor
        sat just outside the window that was cut for the loser."""
        result = registry.retrieve_multi(
            "ALPHA registration deadline",
            subqueries=["ALPHA registration deadline", "BETA appeal deadline"],
            resolve_versions=True, k=8,
        )
        assert "beta-appeal-v2" in result.documents, (
            "BETA's coverage slot was spent on its revoked version and must "
            "be re-reserved once that version is eliminated"
        )
        assert len(result.citations) == 8, "the context stays full"

    def test_an_explicit_status_filter_is_not_overridden(self, registry):
        """Asking for revoked documents and then having them deleted for being
        revoked would undo the caller's own instruction."""
        result = registry.retrieve_multi(
            "what did the BETA appeal rule use to say?",
            subqueries=["BETA appeal deadline"],
            subquery_filters=[{"status": "REVOGADO"}, {"status": "REVOGADO"}],
            resolve_versions=True, k=8,
        )
        assert "beta-appeal-v1" in result.documents

    def test_eliminations_are_recorded_with_a_reason(self, registry):
        result = registry.retrieve_multi(
            "BETA appeal deadline",
            subqueries=["BETA appeal notification", "secretariat deadline"],
            resolve_versions=True, k=8, explain=True,
        )
        reasons = [e["reason"] for e in result.plan["eliminated"]
                   if e.get("document_id") == "beta-appeal-v1"]
        assert reasons and "superseded" in reasons[0]


class TestUndecidableConflict:
    """Two documents equally current, equally versioned, same date, and they
    disagree. Metadata cannot decide — and alphabetical order is not a
    decision."""

    def test_both_sides_survive(self, registry):
        result = registry.retrieve_multi(
            "DELTA eligibility",
            subqueries=["DELTA eligibility months"],
            resolve_versions=True, k=8,
        )
        assert {"delta-elig-a", "delta-elig-b"} <= set(result.documents), (
            "dropping one of two indistinguishable rules hides the conflict "
            "and makes the answer look certain"
        )

    def test_the_conflict_is_reported_as_unresolved(self, registry):
        result = registry.retrieve_multi(
            "DELTA eligibility",
            subqueries=["DELTA eligibility months"],
            resolve_versions=True, k=8,
        )
        conflict = next(c for c in result.conflicts if c["group"] == "delta-elig")
        assert conflict["resolved"] is False
        assert set(conflict["tied"]) == {"delta-elig-a", "delta-elig-b"}

    def test_a_decidable_group_stays_resolved(self, registry):
        result = registry.retrieve_multi(
            "BETA appeal deadline",
            subqueries=["BETA appeal deadline"],
            resolve_versions=True, k=8,
        )
        conflict = next(c for c in result.conflicts
                        if c["group"] == "beta-appeal")
        assert conflict["resolved"] is True
        assert conflict["tied"] == []
        assert conflict["kept"]["document_id"] == "beta-appeal-v2"

    def test_redundant_restatements_also_tie_rather_than_one_winning(
            self, registry):
        """Two documents stating the same current rule are still two
        documents: metadata cannot rank them, so neither is deleted."""
        result = registry.retrieve_multi(
            "ALPHA registration deadline",
            subqueries=["ALPHA registration deadline"],
            resolve_versions=True, k=8,
        )
        conflict = next(c for c in result.conflicts if c["group"] == "alpha-reg")
        assert conflict["resolved"] is False
        assert set(conflict["tied"]) == {"alpha-reg-v3", "alpha-reg-handbook"}
        assert {d["document_id"] for d in conflict["dropped"]} == {
            "alpha-reg-v1", "alpha-reg-v2"
        }

    def test_the_answer_prompt_declares_the_ambiguity(self, registry):
        seen = {}

        def capture(prompt: str) -> str:
            seen["prompt"] = prompt
            return "The sources do not determine the eligibility for DELTA."

        registry.ask_multi("DELTA eligibility", llm=capture,
                           subqueries=["DELTA eligibility months"],
                           resolve_versions=True, k=8)
        assert "delta-elig" in seen["prompt"]
        assert "could not be resolved" in seen["prompt"]


# --- answering: entities, atomicity, citation support ---------------------


class TestMultiEntityAnswer:
    FACETS = [("ALPHA", "registration"), ("BETA", "appeal")]

    def answer(self, registry, **kwargs):
        return registry.ask_multi(
            "What is the registration deadline for ALPHA and the appeal "
            "deadline for BETA?",
            llm=answerer(self.FACETS),
            subqueries=["ALPHA registration deadline", "BETA appeal deadline"],
            facets=["ALPHA registration deadline", "BETA appeal deadline"],
            verify=judge, resolve_versions=True, citations=True, k=8,
            **kwargs,
        )

    def test_each_entity_gets_its_own_rule(self, registry):
        answer = self.answer(registry)
        assert "45 calendar days" in answer.text
        assert "15 business days" in answer.text
        assert "60 calendar days" not in answer.text, "revoked ALPHA rule"
        assert "30 business days" not in answer.text, "revoked BETA rule"

    def test_every_claim_is_atomic_and_separately_judged(self, registry):
        answer = self.answer(registry)
        report = answer.verification
        assert len(report.claims) == 2, [c.claim for c in report.claims]
        for claim, entity in zip(report.claims, ("ALPHA", "BETA")):
            assert entity in claim.claim
            other = "BETA" if entity == "ALPHA" else "ALPHA"
            assert other not in claim.claim, "one claim, one entity"

    def test_each_claim_cites_its_own_entity(self, registry):
        answer = self.answer(registry)
        for claim in answer.verification.claims:
            entity = next(e for e in ENTITIES if e in claim.claim)
            citations = {c.index: c for c in answer.result.citations}
            for index in claim.citations:
                assert citations[index].metadata.get("entity") == entity

    def test_quotes_are_checked_against_the_cited_source(self, registry):
        answer = self.answer(registry)
        for claim in answer.verification.claims:
            if claim.quote:
                assert claim.quote in claim_source_text(answer, claim)
        assert answer.verification.ok is True

    def test_verified_and_complete(self, registry):
        answer = self.answer(registry)
        assert answer.verification.ok is True
        assert answer.verification.complete is True


def claim_source_text(answer, claim) -> str:
    by_index = {c.index: c for c in answer.result.citations}
    return " ".join(by_index[i].text for i in claim.citations)


class TestUndeterminableFacet:
    """A document that names a topic without stating it must not become an
    answer. The safe answer is the honest one."""

    def test_missing_value_is_declared_not_guessed(self, registry):
        answer = registry.ask_multi(
            "For GAMMA, what is the appeal procedure and what is the fee?",
            llm=answerer([("GAMMA", "appeal"), ("GAMMA", "fee")]),
            subqueries=["GAMMA appeal procedure", "GAMMA fee"],
            facets=["GAMMA appeal procedure", "GAMMA fee"],
            verify=judge, resolve_versions=True, k=8,
        )
        assert "do not determine the appeal for GAMMA" in answer.text
        assert "250 credits" in answer.text

    def test_an_undetermined_facet_is_not_a_covered_facet(self, registry):
        answer = registry.ask_multi(
            "For GAMMA, what is the appeal procedure and what is the fee?",
            llm=answerer([("GAMMA", "appeal"), ("GAMMA", "fee")]),
            subqueries=["GAMMA appeal procedure", "GAMMA fee"],
            facets=["GAMMA appeal procedure", "GAMMA fee"],
            verify=judge, resolve_versions=True, k=8,
        )
        assert answer.verification.complete is False
        assert answer.verification.ok is True, (
            "declining to answer is not an unsupported claim"
        )


# --- the adversarial paths ------------------------------------------------


class TestEntityCrossContamination:
    """The failure the corpus is built to produce: a plausible answer that
    attaches one entity's rule to another."""

    def test_a_crossed_entity_is_caught_and_removed(self, registry):
        def crossing_llm(prompt: str) -> str:
            context = prompt.split("# Context\n", 1)[-1]
            beta = next(i for i, b in _blocks(context)
                        if "15 business days" in b)
            return (f"The registration deadline for ALPHA is 15 business "
                    f"days [{beta}].")

        answer = registry.ask_multi(
            "ALPHA registration deadline",
            llm=crossing_llm,
            subqueries=["ALPHA registration deadline", "BETA appeal deadline"],
            verify=judge, verification_mode="repair",
            resolve_versions=True, k=8,
        )
        claim = answer.verification.claims[0]
        assert claim.verdict == "contradicted"
        assert "cites sources about" in claim.rationale
        assert answer.verification.ok is False
        assert "15 business days" not in answer.text

    def test_a_figure_from_a_revoked_rule_cannot_be_smuggled_in(self, registry):
        def stale_llm(prompt: str) -> str:
            context = prompt.split("# Context\n", 1)[-1]
            alpha = next(i for i, b in _blocks(context) if "ALPHA" in b)
            return f"The registration deadline for ALPHA is 90 days [{alpha}]."

        answer = registry.ask_multi(
            "ALPHA registration deadline", llm=stale_llm,
            subqueries=["ALPHA registration deadline"],
            verify=judge, verification_mode="repair",
            resolve_versions=True, k=8,
        )
        assert answer.verification.claims[0].verdict == "unsupported"
        assert "90 days" not in answer.text


class TestUnresolvedConflictReachesTheAnswer:
    """When metadata cannot say which rule applies, both must reach the model
    and the answer must not sound settled."""

    def test_both_figures_are_in_the_context(self, registry):
        result = registry.retrieve_multi(
            "DELTA eligibility", subqueries=["DELTA eligibility months"],
            resolve_versions=True, k=8,
        )
        assert "24 months" in result.context
        assert "36 months" in result.context

    def test_a_one_sided_answer_is_not_certified_complete(self, registry):
        """Picking one of two equally current rules is a choice the sources do
        not support. The judge sees both, so the claim cannot be `supported`
        just because one document happens to contain the figure."""
        def one_sided(prompt: str) -> str:
            context = prompt.split("# Context\n", 1)[-1]
            index = next(i for i, b in _blocks(context) if "24 months" in b)
            return f"An applicant qualifies for DELTA after 24 months [{index}]."

        unresolved_groups = {
            c["group"] for c in registry.retrieve_multi(
                "DELTA eligibility", subqueries=["DELTA eligibility months"],
                resolve_versions=True, k=8,
            ).conflicts if not c["resolved"]
        }
        assert "delta-elig" in unresolved_groups

        def conflict_aware(payload):
            claims = []
            for item in payload["claims"]:
                cited_groups = {
                    e["metadata"].get("doc_group") for e in item["evidence"]
                }
                disputed = bool(cited_groups & unresolved_groups)
                claims.append({
                    "verdict": "unsupported" if disputed else "supported",
                    "rationale": ("the group has equally current documents "
                                  "that disagree; one figure is not the rule"),
                })
            return {"claims": claims,
                    "facets": [{"facet": f, "covered": False,
                                "rationale": "the sources disagree"}
                               for f in payload["facets"]]}

        answer = registry.ask_multi(
            "DELTA eligibility", llm=one_sided,
            subqueries=["DELTA eligibility months"],
            facets=["DELTA eligibility period"],
            verify=conflict_aware, verification_mode="repair",
            resolve_versions=True, k=8,
        )
        assert answer.verification.ok is False
        assert answer.verification.complete is False
        assert "24 months" not in answer.text


class TestQuestionFactRuleAndInference:
    """Three different things that all look like sentences: what the user told
    us, what a document says, and what follows from putting them together."""

    def test_each_is_labelled_and_only_the_rule_needs_a_source(self, registry):
        question = ("I received the ALPHA notice of eligibility on 2025-03-02. "
                    "What is my registration deadline?")

        def reasoning_llm(prompt: str) -> str:
            context = prompt.split("# Context\n", 1)[-1]
            index = next(i for i, b in _blocks(context)
                         if "45 calendar days" in b)
            return (
                "You received the ALPHA notice on 2025-03-02. "
                f"The registration deadline for ALPHA is 45 calendar days "
                f"from the notice of eligibility [{index}]. "
                "Your deadline therefore falls on 2025-04-16."
            )

        def three_way(payload):
            out = []
            for item in payload["claims"]:
                claim = item["claim"]
                if "2025-03-02" in claim:
                    out.append({"verdict": "question_fact",
                                "rationale": "stated in the question"})
                elif "therefore" in claim:
                    out.append({"verdict": "inference",
                                "rationale": "rule applied to the given date"})
                else:
                    out.append({"verdict": "supported",
                                "rationale": "matches the cited rule"})
            return out

        answer = registry.ask_multi(
            question, llm=reasoning_llm,
            subqueries=["ALPHA registration deadline"],
            verify=three_way, verification_mode="strict",
            resolve_versions=True, k=8,
        )
        verdicts = [c.verdict for c in answer.verification.claims]
        assert verdicts == ["question_fact", "supported", "inference"]
        assert answer.verification.ok is True
        # `strict` drops uncited claims, and neither the question fact nor the
        # inference is uncited — they are sourced from the question and from
        # the rule respectively.
        assert "2025-03-02" in answer.text
        assert "2025-04-16" in answer.text


class TestClaimAtomicity:
    def test_two_rules_in_one_sentence_are_judged_separately(self, registry):
        def run_on(prompt: str) -> str:
            context = prompt.split("# Context\n", 1)[-1]
            alpha = next(i for i, b in _blocks(context)
                         if "45 calendar days" in b)
            beta = next(i for i, b in _blocks(context)
                        if "15 business days" in b)
            return (f"ALPHA registration closes after 45 calendar days "
                    f"[{alpha}] and BETA appeals close after 5 business days "
                    f"[{beta}].")

        def resegmenting(payload):
            assert len(payload["claims"]) == 1, "one sentence, two rules"
            answer = payload["answer"]
            first, second = answer.split(" and ", 1)
            return [
                {"claim": first, "verdict": "supported",
                 "rationale": "matches the ALPHA rule"},
                {"claim": second, "verdict": "unsupported",
                 "rationale": "the BETA rule says 15, not 5"},
            ]

        answer = registry.ask_multi(
            "ALPHA registration and BETA appeal deadlines", llm=run_on,
            subqueries=["ALPHA registration deadline", "BETA appeal deadline"],
            verify=resegmenting, verification_mode="repair",
            resolve_versions=True, k=8,
        )
        report = answer.verification
        assert report.segmentation == "verifier"
        assert len(report.claims) == 2
        assert report.ok is False
        assert "45 calendar days" in answer.text, "the correct half survives"
        assert "5 business days" not in answer.text


class TestFailClosedUnderLoad:
    def test_a_judge_that_dies_mid_corpus_does_not_bless_the_answer(
            self, registry):
        def dying(payload):
            raise RuntimeError("judge unavailable")

        answer = registry.ask_multi(
            "ALPHA registration deadline",
            llm=answerer([("ALPHA", "registration")]),
            subqueries=["ALPHA registration deadline"],
            facets=["ALPHA registration deadline"],
            verify=dying, verification_mode="repair", resolve_versions=True,
            k=8,
        )
        assert answer.verification.ok is False
        assert answer.verification.complete is False
        assert "45 calendar days" in answer.text, "answer preserved intact"

    def test_a_judge_that_skips_a_claim_fails_closed(self, registry):
        def partial(payload):
            return [{"verdict": "supported"}]

        answer = registry.ask_multi(
            "ALPHA registration and BETA appeal",
            llm=answerer([("ALPHA", "registration"), ("BETA", "appeal")]),
            subqueries=["ALPHA registration deadline", "BETA appeal deadline"],
            facets=["ALPHA registration deadline", "BETA appeal deadline"],
            verify=partial, resolve_versions=True, k=8,
        )
        assert answer.verification.ok is False
        assert answer.verification.complete is False

    def test_invented_citation_markers_never_survive(self, registry):
        answer = registry.ask_multi(
            "ALPHA registration deadline",
            llm=lambda p: "The deadline is 45 calendar days [1] and [99].",
            subqueries=["ALPHA registration deadline"],
            resolve_versions=True, k=8,
        )
        assert "[99]" not in answer.text
