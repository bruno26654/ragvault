"""Post-generation semantic validation for ask() and ask_multi().

The hard cases: a citation that exists but does not support the claim, a
premise from the user's question dressed up as documented evidence, a claim
citing a revoked version, and a claim with no citation at all.
"""

from __future__ import annotations

import pytest

import ragvault


@pytest.fixture
def kb(tmp_path):
    base = ragvault.open(tmp_path / "kb")
    base.add([
        {"id": "refund", "text":
         "Refund requests must be filed within 30 days of purchase.",
         "metadata": {"status": "VIGENTE", "doc_group": "refund",
                      "effective_date": "2024-01-01", "version": 2}},
        {"id": "refund_old", "text":
         "Refund requests must be filed within 90 days of purchase.",
         "metadata": {"status": "REVOGADO", "doc_group": "refund",
                      "effective_date": "2019-01-01", "version": 1}},
        {"id": "shipping", "text":
         "Orders ship within five business days nationwide."},
    ])
    yield base
    base.close()


def verdicts(*items):
    """Build a verifier returning fixed verdicts, in claim order."""
    def _verify(payload):
        assert len(payload["claims"]) == len(items), (
            f"expected {len(items)} claims, got {len(payload['claims'])}: "
            f"{[c['claim'] for c in payload['claims']]}"
        )
        return list(items)
    return _verify


class TestWrongButExistingCitation:
    """The failure citation-sanitizing cannot catch: [n] is real, but the
    source does not say what the claim says."""

    def test_repair_removes_the_claim(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1]. Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "supported", "rationale": "matches source"},
                {"verdict": "contradicted",
                 "rationale": "[1] says 30 days, not instant"},
            ),
            verification_mode="repair",
            k=3,
        )
        assert "instant" not in answer.text
        assert "30 days [1]" in answer.text
        assert not answer.verification.ok
        assert answer.verification.claims[1].action == "removed"

    def test_repair_uses_a_replacement_when_offered(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts({
                "verdict": "contradicted",
                "rationale": "source says 30 days",
                "replacement": "Refunds must be requested within 30 days [1].",
            }),
            verification_mode="repair",
            k=3,
        )
        assert answer.text == "Refunds must be requested within 30 days [1]."
        assert answer.verification.claims[0].action == "rewritten"

    def test_annotate_marks_without_removing(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts({"verdict": "contradicted", "rationale": "no"}),
            verification_mode="annotate",
            k=3,
        )
        assert "instant" in answer.text
        assert "[contradicted]" in answer.text
        assert answer.verification.claims[0].action == "annotated"

    def test_report_changes_nothing(self, kb):
        original = "Refunds are instant [1]."
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: original,
            verify=verdicts({"verdict": "unsupported", "rationale": "no"}),
            verification_mode="report",
            k=3,
        )
        assert answer.text == original
        assert not answer.verification.ok
        assert answer.verification.claims[0].action == "kept"

    def test_evidence_resolves_to_real_chunks(self, kb):
        seen = {}

        def capture(payload):
            seen["payload"] = payload
            return ["supported"]

        kb.ask("refund deadline", llm=lambda p: "Refunds take 30 days [1].",
               verify=capture, k=3)
        evidence = seen["payload"]["claims"][0]["evidence"]
        assert evidence, "the [1] marker must resolve to a real citation"
        chunk_ids = evidence[0]["chunk_ids"]
        assert chunk_ids
        assert kb._vault.get_chunks(chunk_ids)[0] is not None


class TestQuestionFactVsDocumentEvidence:
    def test_question_fact_is_distinguished_from_evidence(self, kb):
        answer = kb.ask(
            "I bought this on March 1st — what is the refund deadline?",
            llm=lambda p: ("You bought it on March 1st [1]. "
                           "Refunds must be filed within 30 days [1]."),
            verify=verdicts(
                {"verdict": "question_fact",
                 "rationale": "the purchase date comes from the question, "
                              "not from [1]"},
                {"verdict": "supported", "rationale": "matches [1]"},
            ),
            verification_mode="report",
            k=3,
        )
        kinds = [c.verdict for c in answer.verification.claims]
        assert kinds == ["question_fact", "supported"]
        # question_fact is not an error: the answer stays intact
        assert answer.verification.ok

    def test_inference_is_its_own_verdict(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "So you should act quickly [1].",
            verify=verdicts({"verdict": "inference", "rationale": "derived"}),
            k=3,
        )
        assert answer.verification.claims[0].verdict == "inference"
        assert answer.verification.ok


class TestVersionConflict:
    def test_claim_citing_a_revoked_version_is_repaired(self, kb):
        """resolve_versions keeps the revoked doc out of the context; if the
        model still asserts the old rule, verification catches it."""
        answer = kb.ask_multi(
            "refund deadline",
            llm=lambda p: "Refunds must be filed within 90 days [1].",
            verify=verdicts({
                "verdict": "contradicted",
                "rationale": "the current version says 30 days; 90 days is the "
                             "revoked version",
                "replacement": "Refunds must be filed within 30 days [1].",
            }),
            verification_mode="repair",
            resolve_versions=True,
            k=5,
        )
        assert "30 days" in answer.text
        assert "90 days" not in answer.text
        # the revoked document was never in the context to begin with
        assert "refund_old" not in answer.result.documents

    def test_conflicts_are_visible_alongside_verification(self, kb):
        answer = kb.ask_multi(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            verify=verdicts("supported"),
            resolve_versions=True,
            k=5,
        )
        assert answer.result.conflicts
        assert answer.verification.ok


class TestUncitedClaims:
    def test_strict_drops_uncited_claims(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1]. Everyone loves refunds.",
            verify=verdicts("supported", "uncited"),
            verification_mode="strict",
            k=3,
        )
        assert "Everyone loves refunds" not in answer.text
        assert "30 days [1]" in answer.text
        assert answer.verification.claims[1].action == "removed"

    def test_repair_keeps_uncited_claims(self, kb):
        """`repair` targets wrong claims; `strict` also targets uncited ones."""
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1]. Everyone loves refunds.",
            verify=verdicts("supported", "uncited"),
            verification_mode="repair",
            k=3,
        )
        assert "Everyone loves refunds" in answer.text

    def test_claim_without_citation_has_empty_evidence(self, kb):
        seen = {}

        def capture(payload):
            seen["p"] = payload
            return ["uncited"]

        kb.ask("refund", llm=lambda p: "No citation here.", verify=capture, k=3)
        assert seen["p"]["claims"][0]["citations"] == []
        assert seen["p"]["claims"][0]["evidence"] == []

    def test_everything_removed_says_so(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts("contradicted"),
            verification_mode="repair",
            k=3,
        )
        assert answer.text, "an empty answer would hide the failure"
        assert "no statement" in answer.text.lower()
        assert "supported by the retrieved context" in answer.text.lower()


class TestVerifierFailure:
    def test_exception_preserves_the_original_answer(self, kb):
        def exploding(payload):
            raise RuntimeError("verifier offline")

        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            verify=exploding, verification_mode="repair", k=3,
        )
        assert answer.text == "Refunds take 30 days [1]."
        assert "verifier offline" in answer.verification.error
        assert answer.verification.claims == []

    def test_wrong_verdict_count_preserves_the_answer(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "One [1]. Two [1].",
            verify=lambda payload: ["supported"],  # only 1 for 2 claims
            verification_mode="repair", k=3,
        )
        assert answer.text == "One [1]. Two [1]."
        assert "2 claims" in answer.verification.error

    def test_verifier_returning_none_preserves_the_answer(self, kb):
        answer = kb.ask(
            "refund deadline", llm=lambda p: "Refunds take 30 days [1].",
            verify=lambda payload: None, verification_mode="repair", k=3,
        )
        assert answer.text == "Refunds take 30 days [1]."
        assert answer.verification.error

    def test_unknown_verdict_is_an_actionable_error(self, kb):
        with pytest.raises(ragvault.ConfigurationError) as err:
            kb.ask("refund", llm=lambda p: "Claim [1].",
                   verify=verdicts("probably-fine"), k=3)
        assert "unknown verdict" in str(err.value)

    def test_unknown_mode_is_rejected(self, kb):
        with pytest.raises(ragvault.ConfigurationError) as err:
            kb.ask("refund", llm=lambda p: "Claim [1].",
                   verify=verdicts("supported"), verification_mode="magic", k=3)
        assert "verification_mode" in str(err.value)

    def test_non_callable_verifier_is_rejected(self, kb):
        with pytest.raises(ragvault.ConfigurationError):
            kb.ask("refund", llm=lambda p: "Claim [1].", verify="nope", k=3)


class TestTraceAndReport:
    def test_trace_records_the_full_verification(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1]. Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "supported", "rationale": "matches"},
                {"verdict": "contradicted", "rationale": "conflicts with [1]"},
            ),
            verification_mode="repair", trace=True, k=3,
        )
        v = answer.result.trace["verification"]
        assert v["mode"] == "repair"
        assert v["ok"] is False
        assert v["counts"] == {"supported": 1, "contradicted": 1}
        assert v["elapsed_ms"] >= 0
        first, second = v["claims"]
        assert first["verdict"] == "supported" and first["action"] == "kept"
        assert second["rationale"] == "conflicts with [1]"
        assert second["action"] == "removed"
        assert first["citations"] == [1]
        assert first["chunk_ids"]

    def test_multi_query_trace_carries_verification(self, kb):
        answer = kb.ask_multi(
            "refund deadline and shipping",
            subqueries=["refund deadline", "shipping time"],
            llm=lambda p: "Refunds take 30 days [1].",
            verify=verdicts("supported"),
            trace=True, k=5,
        )
        assert answer.result.trace["verification"]["ok"] is True
        assert answer.result.trace["subqueries"]

    def test_report_helpers(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "A [1]. B [1].",
            verify=verdicts("supported", "unsupported"),
            verification_mode="report", k=3,
        )
        report = answer.verification
        assert len(report.unsupported) == 1
        assert answer.unverified_claims == report.unsupported
        assert "issues" in repr(answer)
        assert "VerificationReport" in repr(report)


class TestCompatibility:
    def test_ask_without_verify_is_unchanged(self, kb):
        answer = kb.ask("refund deadline", llm=lambda p: "30 days [1]", k=3)
        assert answer.text == "30 days [1]"
        assert answer.verification is None
        assert answer.unverified_claims == []

    def test_ask_multi_without_verify_is_unchanged(self, kb):
        answer = kb.ask_multi("refund deadline", llm=lambda p: "30 days [1]", k=3)
        assert answer.text == "30 days [1]"
        assert answer.verification is None

    def test_citation_sanitizing_still_runs_before_verification(self, kb):
        """ask_multi strips invented markers first; the verifier then sees the
        cleaned text."""
        seen = {}

        def capture(payload):
            seen["answer"] = payload["answer"]
            return ["supported"]

        answer = kb.ask_multi(
            "refund", llm=lambda p: "Refunds take 30 days [1] and [99].",
            verify=capture, k=3,
        )
        assert "[99]" not in seen["answer"]
        assert "[99]" not in answer.text

    def test_verification_survives_a_single_sentence_answer(self, kb):
        answer = kb.ask("refund", llm=lambda p: "Just one claim [1]",
                        verify=verdicts("supported"), k=3)
        assert len(answer.verification.claims) == 1

    def test_empty_answer_yields_no_claims(self, kb):
        answer = kb.ask("refund", llm=lambda p: "   ",
                        verify=lambda payload: [], k=3)
        assert answer.verification.claims == []
        assert answer.verification.ok


class TestClaimSplitting:
    def test_splits_on_sentence_boundaries_keeping_markers(self):
        from ragvault.verification import citations_in, split_claims

        claims = split_claims("First claim [1]. Second claim [2]! Third [3]?")
        assert len(claims) == 3
        assert citations_in(claims[1]) == [2]

    def test_repeated_marker_is_listed_once(self):
        from ragvault.verification import citations_in

        assert citations_in("A [2] and again [2] and [3].") == [2, 3]
