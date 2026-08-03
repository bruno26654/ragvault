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


def verdicts(*items, recheck="supported"):
    """Build a verifier returning fixed verdicts, in claim order.

    `repair`/`strict` re-verify the replacements they applied, so the stub
    answers that second pass separately: a coherent verifier stands by the
    correction it proposed. Pass ``recheck=`` to simulate one that does not.
    """
    state = {"first": True}

    def _verify(payload):
        if state["first"]:
            state["first"] = False
            assert len(payload["claims"]) == len(items), (
                f"expected {len(items)} claims, got {len(payload['claims'])}: "
                f"{[c['claim'] for c in payload['claims']]}"
            )
            return list(items)
        # second pass: judge only the replacements
        return [recheck] * len(payload["claims"])
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

    def test_replacements_are_ignored_by_default(self, kb):
        """The verifier segments and classifies; it does not write. Grading
        its own replacement would be self-endorsement, not verification."""
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
        assert "Refunds must be requested" not in answer.text
        assert answer.verification.claims[0].action == "removed"
        assert not answer.verification.ok

    def test_repair_uses_a_replacement_when_opted_in(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts({
                "verdict": "contradicted",
                "rationale": "source says 30 days",
                "replacement": "Refunds must be requested within 30 days [1].",
            }),
            verification_mode="repair",
            allow_replacements=True,
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
            allow_replacements=True,
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


class TestFormattingPreserved:
    """A repair must not flatten the answer's layout: dropping one bullet
    cannot turn a list into a run-on paragraph."""

    def test_bullet_list_survives_a_removal(self, kb):
        answer = kb.ask(
            "refund rules",
            llm=lambda p: ("- Refunds take 30 days [1].\n"
                           "- Refunds are instant [1].\n"
                           "- Shipping takes five days [1]."),
            verify=verdicts("supported", "contradicted", "supported"),
            verification_mode="repair", k=3,
        )
        assert "\n" in answer.text, f"layout was flattened: {answer.text!r}"
        assert answer.text.count("- ") == 2
        assert "instant" not in answer.text

    def test_paragraph_breaks_survive(self, kb):
        answer = kb.ask(
            "refund rules",
            llm=lambda p: "First para [1].\n\nSecond para [1].\n\nThird para [1].",
            verify=verdicts("supported", "contradicted", "supported"),
            verification_mode="repair", k=3,
        )
        assert "\n\n" in answer.text
        assert "Second para" not in answer.text

    def test_single_space_answers_are_unaffected(self, kb):
        answer = kb.ask(
            "refund rules",
            llm=lambda p: "One [1]. Two [1]. Three [1].",
            verify=verdicts("supported", "contradicted", "supported"),
            verification_mode="repair", k=3,
        )
        assert answer.text == "One [1]. Three [1]."

    def test_split_preserves_the_original_text(self):
        from ragvault.verification import _split_with_separators

        text = "- A one.\n- B two.\n\n- C three."
        claims, seps = _split_with_separators(text)
        rebuilt = claims[0] + "".join(s + c for s, c in zip(seps, claims[1:]))
        assert rebuilt == text.strip()


class TestReplacementRecheck:
    """A `replacement` is generated text that would otherwise enter the answer
    unchecked — the repair itself can introduce an inaccuracy. Replacements are
    opt-in (`allow_replacements=True`); these tests exercise that path."""

    def test_rejected_replacement_is_dropped(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "contradicted", "rationale": "wrong",
                 "replacement": "Refunds are also instant, honestly [1]."},
                recheck="unsupported",   # verificador rejeita a própria correção
            ),
            verification_mode="repair", allow_replacements=True, k=3,
        )
        assert "instant" not in answer.text
        claim = answer.verification.claims[0]
        assert claim.replacement_verdict == "unsupported"
        assert claim.action == "removed"
        assert "replacement not accepted" in claim.rationale

    def test_accepted_replacement_is_kept(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "contradicted", "rationale": "wrong",
                 "replacement": "Refunds must be filed within 30 days [1]."},
                recheck="supported",
            ),
            verification_mode="repair", allow_replacements=True, k=3,
        )
        assert answer.text == "Refunds must be filed within 30 days [1]."
        assert answer.verification.claims[0].replacement_verdict == "supported"
        assert answer.verification.claims[0].action == "rewritten"

    def test_recheck_runs_only_once_no_loop(self, kb):
        calls = {"n": 0}

        def counting(payload):
            calls["n"] += 1
            return [{"verdict": "contradicted", "rationale": "no",
                     "replacement": "Another attempt [1]."}
                    ] * len(payload["claims"])

        kb.ask("refund", llm=lambda p: "Bad claim [1].",
               verify=counting, verification_mode="repair",
               allow_replacements=True, k=3)
        assert calls["n"] == 2, "exactly one extra pass, never a loop"

    def test_failing_recheck_keeps_the_repair_and_says_so(self, kb):
        state = {"first": True}

        def flaky(payload):
            if state["first"]:
                state["first"] = False
                return [{"verdict": "contradicted", "rationale": "wrong",
                         "replacement": "Corrected text [1]."}]
            raise RuntimeError("judge offline")

        answer = kb.ask("refund", llm=lambda p: "Bad claim [1].",
                        verify=flaky, verification_mode="repair",
                        allow_replacements=True, k=3)
        assert answer.text == "Corrected text [1]."
        assert "judge offline" in answer.verification.recheck_error

    def test_no_recheck_without_replacements(self, kb):
        calls = {"n": 0}

        def counting(payload):
            calls["n"] += 1
            return ["supported"] * len(payload["claims"])

        kb.ask("refund", llm=lambda p: "Fine claim [1].",
               verify=counting, verification_mode="repair", k=3)
        assert calls["n"] == 1


class TestFacetCoverage:
    """Fidelity and completeness are different axes: every claim can be
    supported and the answer still miss a facet entirely."""

    def test_facets_reach_the_verifier(self, kb):
        seen = {}

        def capture(payload):
            seen["facets"] = payload["facets"]
            return ["supported"]

        kb.ask_multi("refund deadline and payment", llm=lambda p: "30 days [1].",
                     subqueries=["refund deadline", "refund payment"],
                     verify=capture, k=5)
        assert seen["facets"] == ["refund deadline", "refund payment"]

    def test_uncovered_facet_is_reported_without_touching_the_text(self, kb):
        def verifier(payload):
            return {
                "claims": ["supported"] * len(payload["claims"]),
                "facets": [
                    {"facet": "refund deadline", "covered": True},
                    {"facet": "refund payment", "covered": False,
                     "rationale": "a resposta não trata do pagamento"},
                ],
            }

        answer = kb.ask_multi(
            "refund deadline and payment", llm=lambda p: "30 days [1].",
            subqueries=["refund deadline", "refund payment"],
            verify=verifier, verification_mode="repair", k=5,
        )
        # fidelity is fine, completeness is not — and repair must not invent
        assert answer.verification.ok
        assert answer.verification.complete is False
        assert answer.text == "30 days [1]."
        assert answer.verification.uncovered_facets[0]["facet"] == "refund payment"

    def test_complete_is_none_when_not_reported(self, kb):
        answer = kb.ask_multi(
            "refund deadline", llm=lambda p: "30 days [1].",
            subqueries=["refund deadline"], verify=verdicts("supported"), k=5,
        )
        assert answer.verification.complete is None, (
            "no report is not the same as full coverage"
        )

    def test_coverage_lands_in_the_trace(self, kb):
        def verifier(payload):
            return {"claims": ["supported"] * len(payload["claims"]),
                    "facets": [{"facet": "refund deadline", "covered": True}]}

        answer = kb.ask_multi(
            "refund rules", llm=lambda p: "30 days [1].",
            subqueries=["refund deadline"], verify=verifier, trace=True, k=5,
        )
        v = answer.result.trace["verification"]
        assert v["complete"] is True
        assert v["facet_coverage"][0]["facet"] == "refund deadline"

    def test_subquery_equal_to_the_question_is_not_a_facet(self, kb):
        """Dedup drops a subquery identical to the question, so there is no
        facet left to report coverage on."""
        seen = {}

        def capture(payload):
            seen["facets"] = payload["facets"]
            return ["supported"]

        kb.ask_multi("refund deadline", llm=lambda p: "30 days [1].",
                     subqueries=["refund deadline"], verify=capture, k=5)
        assert seen["facets"] == []


class TestEvidenceMetadata:
    def test_precedence_fields_travel_with_the_evidence(self, kb):
        seen = {}

        def capture(payload):
            seen["p"] = payload
            return ["supported"]

        kb.ask("refund deadline", llm=lambda p: "30 days [1].",
               verify=capture, k=3)
        meta = seen["p"]["claims"][0]["evidence"][0]["metadata"]
        assert meta["status"] == "VIGENTE"
        assert meta["doc_group"] == "refund"
        assert meta["version"] == 2

    def test_citations_expose_metadata(self, kb):
        result = kb.retrieve("refund deadline", k=3)
        assert result.citations[0].metadata
        assert "status" in result.citations[0].to_dict()["metadata"]


class TestClaimBoundaries:
    """Heuristic boundaries, including the scripts and abbreviations the
    original rule silently ignored."""

    def _split(self, text):
        from ragvault.verification import _split_with_separators
        return _split_with_separators(text)

    @pytest.mark.parametrize("text,expected", [
        ("Um [1]. Dois [2]. Tres [3].", 3),
        ("- A um [1].\n- B dois [2].", 2),
        ("O valor é 3.14 reais [1]. Outro ponto [2].", 2),      # decimal
        ("O Art. 5º define o prazo [1]. Ja o Inc. II trata [2].", 2),  # abbrev
        ("退款需要30天[1]。运输需要5天[2]。", 2),                    # chinês
        ("返金には30日かかります[1]。配送には5日かかります[2]。", 2),   # japonês
        ("يستغرق الاسترداد 30 يومًا [1]. يستغرق الشحن 5 أيام [2].", 2),  # árabe
        ("ההחזר אורך 30 יום [1]. המשלוח אורך 5 ימים [2].", 2),      # hebraico
    ])
    def test_boundaries(self, text, expected):
        assert len(self._split(text)[0]) == expected

    @pytest.mark.parametrize("text", [
        "Um [1]. Dois [2].",
        "- A [1].\n- B [2].",
        "退款需要30天[1]。运输需要5天[2]。",
        "O Art. 5º vale [1]. Fim [2].",
    ])
    def test_split_is_lossless(self, text):
        claims, seps = self._split(text)
        rebuilt = claims[0] + "".join(s + c for s, c in zip(seps, claims[1:]))
        assert rebuilt == text.strip()

    def test_cjk_answer_can_be_repaired_per_claim(self, kb):
        """Before, a CJK answer was one claim: verification was a no-op."""
        answer = kb.ask(
            "refund",
            llm=lambda p: "退款需要30天[1]。运输需要5天[1]。",
            verify=verdicts("supported", "contradicted"),
            verification_mode="repair", k=3,
        )
        assert "退款需要30天" in answer.text
        assert "运输需要5天" not in answer.text


class TestVerifierSegmentation:
    """The heuristic cannot see two claims inside one sentence; the verifier
    is already an LLM reading the answer, so its segmentation is free."""

    def test_verifier_can_split_one_sentence_into_two_claims(self, kb):
        sentence = "Refunds take 30 days [1] and shipping takes 5 days [1]."

        def segmenting(payload):
            # ignores the heuristic single claim and returns two
            return [
                {"claim": "Refunds take 30 days [1]", "verdict": "supported",
                 "rationale": "matches"},
                {"claim": "shipping takes 5 days [1]", "verdict": "contradicted",
                 "rationale": "the source says nothing about shipping"},
            ]

        answer = kb.ask("refund", llm=lambda p: sentence, verify=segmenting,
                        verification_mode="repair", k=3)
        assert answer.verification.segmentation == "verifier"
        assert len(answer.verification.claims) == 2
        # only the offending half is gone — the correct half survives
        assert "Refunds take 30 days [1]" in answer.text
        assert "shipping takes 5 days" not in answer.text

    def test_heuristic_alone_would_drop_the_whole_sentence(self, kb):
        """Contrast: without re-segmentation the sentence is one claim, so a
        single bad half condemns the correct half too."""
        sentence = "Refunds take 30 days [1] and shipping takes 5 days [1]."
        answer = kb.ask("refund", llm=lambda p: sentence,
                        verify=verdicts("contradicted"),
                        verification_mode="repair", k=3)
        assert answer.verification.segmentation == "heuristic"
        assert "Refunds take 30 days" not in answer.text

    def test_non_verbatim_claims_are_rejected(self, kb):
        """A re-segmentation that paraphrases would make repair rewrite the
        answer from model output instead of cutting the original."""
        def paraphrasing(payload):
            return [{"claim": "Something the answer never said",
                     "verdict": "supported"}]

        answer = kb.ask("refund", llm=lambda p: "Refunds take 30 days [1].",
                        verify=paraphrasing, verification_mode="repair", k=3)
        assert answer.text == "Refunds take 30 days [1]."
        assert "not verbatim substrings" in answer.verification.error

    def test_segmentation_defaults_to_heuristic(self, kb):
        answer = kb.ask("refund", llm=lambda p: "Refunds take 30 days [1].",
                        verify=verdicts("supported"), k=3)
        assert answer.verification.segmentation == "heuristic"

    def test_resegmentation_preserves_layout(self, kb):
        """A re-segmentation that groups two bullets into one claim must keep
        the newline between them when that claim is kept."""
        def segmenting(payload):
            return [
                {"claim": "- A one [1].\n- B two [1].", "verdict": "supported"},
                {"claim": "- C three [1].", "verdict": "contradicted"},
            ]

        answer = kb.ask(
            "refund", llm=lambda p: "- A one [1].\n- B two [1].\n- C three [1].",
            verify=segmenting, verification_mode="repair", k=3,
        )
        assert answer.verification.segmentation == "verifier"
        assert answer.text == "- A one [1].\n- B two [1]."

    def test_identical_segmentation_stays_heuristic(self, kb):
        """Returning the same claims is not a re-segmentation — no false
        signal in the report."""
        def echoing(payload):
            return [{"claim": c["claim"], "verdict": "supported"}
                    for c in payload["claims"]]

        answer = kb.ask("refund", llm=lambda p: "One [1]. Two [1].",
                        verify=echoing, k=3)
        assert answer.verification.segmentation == "heuristic"
        assert len(answer.verification.claims) == 2

    def test_segmentation_appears_in_the_trace(self, kb):
        def segmenting(payload):
            return [{"claim": "Refunds take 30 days [1].", "verdict": "supported"}]

        answer = kb.ask("refund", llm=lambda p: "Refunds take 30 days [1].",
                        verify=segmenting, trace=True, k=3)
        seg = answer.result.trace["verification"]["segmentation"]
        assert seg in ("heuristic", "verifier")


class TestFailsClosed:
    """A safety feature that fails *open* is worse than none: anything gating
    on `ok`/`complete` would ship unverified text believing it was checked."""

    def test_crashed_verifier_is_not_ok(self, kb):
        def exploding(payload):
            raise RuntimeError("judge offline")

        answer = kb.ask("refund", llm=lambda p: "Refunds are instant [1].",
                        verify=exploding, k=3)
        assert answer.verification.ok is False, (
            "no claims were judged, so nothing may be reported as faithful"
        )
        assert answer.verification.valid is False
        assert answer.text == "Refunds are instant [1].", "answer preserved"

    def test_partial_facet_report_is_not_complete(self, kb):
        def partial(payload):
            return {"claims": ["supported"] * len(payload["claims"]),
                    "facets": [{"facet": payload["facets"][0], "covered": True}]}

        answer = kb.ask_multi(
            "refund deadline and payment method",
            subqueries=["refund deadline", "refund payment method"],
            llm=lambda p: "30 days [1].", verify=partial, k=5,
        )
        assert answer.verification.complete is False
        assert [f["facet"] for f in answer.verification.uncovered_facets] == [
            "refund payment method"
        ]
        assert answer.verification.structural_issues

    def test_unreported_facet_counts_as_uncovered(self, kb):
        def none_reported(payload):
            return ["supported"] * len(payload["claims"])

        answer = kb.ask_multi(
            "refund deadline and payment",
            subqueries=["refund deadline", "refund payment"],
            llm=lambda p: "30 days [1].", verify=none_reported, k=5,
        )
        report = answer.verification
        # Facets were declared and none was evaluated. That is not "unknown":
        # nothing shows the answer is complete, so it fails closed.
        assert report.complete is False
        assert [f["facet"] for f in report.uncovered_facets] == [
            "refund deadline", "refund payment"
        ]
        assert all("did not report" in f["rationale"]
                   for f in report.uncovered_facets)
        # Fidelity is a separate axis: a verifier that simply does not
        # implement the facet protocol still judged every claim.
        assert report.ok

    def test_no_facets_means_unknown_not_complete(self, kb):
        answer = kb.ask("refund", llm=lambda p: "30 days [1].",
                        verify=verdicts("supported"), k=3)
        assert answer.verification.complete is None

    def test_structural_issue_forces_ok_false(self, kb):
        """Even with every verdict 'supported', a structurally invalid result
        cannot be called faithful."""
        def leaving_text_unverified(payload):
            # segments only the first sentence; the second goes unjudged
            return [{"claim": "One [1].", "verdict": "supported"}]

        answer = kb.ask("refund", llm=lambda p: "One [1]. Two [1].",
                        verify=leaving_text_unverified, k=3)
        assert answer.verification.structural_issues
        assert answer.verification.ok is False
        assert "unverified" in answer.verification.structural_issues[0]

    def test_valid_and_ok_when_everything_is_reported(self, kb):
        def complete_verifier(payload):
            return {"claims": ["supported"] * len(payload["claims"]),
                    "facets": [{"facet": f, "covered": True}
                               for f in payload["facets"]]}

        answer = kb.ask_multi(
            "refund deadline and payment",
            subqueries=["refund deadline", "refund payment"],
            llm=lambda p: "30 days [1].", verify=complete_verifier, k=5,
        )
        assert answer.verification.valid is True
        assert answer.verification.ok is True
        assert answer.verification.complete is True


class TestCitedEvidenceIsAtHand:
    """The judge must be able to read the source a claim actually named,
    without re-parsing the assembled context to find block `[n]`."""

    def test_evidence_carries_the_cited_block_text(self, kb):
        seen = {}

        def capture(payload):
            seen.update(payload)
            return ["supported"]

        kb.ask("refund deadline", llm=lambda p: "Refunds take 30 days [1].",
               verify=capture, k=3)
        evidence = seen["claims"][0]["evidence"][0]
        assert "30 days" in evidence["text"]
        assert evidence["text"] in seen["context"], (
            "the evidence is the block itself, not a paraphrase of it"
        )

    def test_citation_text_matches_the_context_block(self, kb):
        result = kb.retrieve("refund deadline", k=3)
        for citation in result.citations:
            assert citation.text, "every citation must carry its own text"
            assert citation.text in result.context, (
                "the citation's text is the block the model was shown"
            )


class TestQuotedEvidence:
    """A quote is the one part of a verdict the library can check itself:
    either the cited source contains those words or it does not."""

    def test_fabricated_quote_cannot_support_a_claim(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts({"verdict": "supported", "rationale": "see source",
                             "quote": "refunds are processed instantly"}),
            verification_mode="repair", k=3,
        )
        claim = answer.verification.claims[0]
        assert claim.verdict == "unsupported"
        assert "does not appear in the cited source" in claim.rationale
        assert answer.verification.ok is False
        assert answer.verification.structural_issues
        assert "instant" not in answer.text

    def test_real_quote_is_accepted(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            verify=verdicts({"verdict": "supported", "rationale": "see source",
                             "quote": "filed within 30 days"}),
            verification_mode="repair", k=3,
        )
        assert answer.verification.ok is True
        assert answer.verification.claims[0].quote == "filed within 30 days"
        assert answer.text == "Refunds take 30 days [1]."

    def test_quote_survives_rewrapping_and_case(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            verify=verdicts({"verdict": "supported",
                             "quote": "Filed   Within\n30 DAYS"}),
            k=3,
        )
        assert answer.verification.ok is True, (
            "whitespace and case do not change whose words they are"
        )

    def test_quote_must_come_from_a_cited_source(self, kb):
        """The text exists in the context — under a block this claim never
        cited. Support has to come from the source the claim named."""
        answer = kb.ask(
            "refund and shipping",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts({"verdict": "supported",
                             "quote": "Orders ship within five business days"}),
            k=3,
        )
        assert answer.verification.claims[0].verdict == "unsupported"

    def test_quotes_are_optional_by_default(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            verify=verdicts("supported"), k=3,
        )
        assert answer.verification.ok is True
        assert answer.verification.claims[0].quote is None

    def test_require_quotes_rejects_silent_support(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            verify=verdicts("supported"), require_quotes=True, k=3,
        )
        assert answer.verification.claims[0].verdict == "unsupported"
        assert answer.verification.ok is False
        assert "quotes are required" in answer.verification.claims[0].rationale

    def test_require_quotes_accepts_a_grounded_span(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            verify=verdicts({"verdict": "supported",
                             "quote": "within 30 days of purchase"}),
            require_quotes=True, k=3,
        )
        assert answer.verification.ok is True


class TestTrailingCitationMarkers:
    """`... 30 days. [1]` is the same citation as `... 30 days [1].` — the
    marker sources the claim it follows, not the one after it."""

    def test_marker_after_the_terminator_stays_with_its_claim(self):
        from ragvault.verification import citations_in, split_claims

        claims = split_claims("Refunds take 30 days. [1] They ship fast. [2]")
        assert claims == ["Refunds take 30 days. [1]", "They ship fast. [2]"]
        assert [citations_in(c) for c in claims] == [[1], [2]]

    def test_marker_without_a_space_still_splits(self):
        from ragvault.verification import citations_in, split_claims

        claims = split_claims("Refunds take 30 days.[1] They ship fast.[2]")
        assert [citations_in(c) for c in claims] == [[1], [2]]

    def test_a_marker_is_never_a_claim_of_its_own(self):
        from ragvault.verification import split_claims

        for text in ("One. [1] Two. [2]", "One.[1] Two.[2]",
                     "First. [1] [2] Second. [3]"):
            assert all(c.strip("[]0123456789 ") for c in split_claims(text)), (
                f"a claim that is only a marker came out of {text!r}"
            )

    def test_marker_on_the_next_line_belongs_to_that_line(self):
        from ragvault.verification import citations_in, split_claims

        claims = split_claims("A claim.\n[1] Next line claim.")
        assert citations_in(claims[0]) == []
        assert citations_in(claims[1]) == [1]

    def test_split_stays_lossless(self):
        from ragvault.verification import _split_with_separators

        for text in ("One. [1] Two. [2]", "One.[1] Two.[2]",
                     "- A. [1]\n- B. [2]", "退款需在30天内申请。[1] 发货需五天。[2]"):
            claims, seps = _split_with_separators(text)
            rebuilt = "".join(
                claim + (seps[i] if i < len(seps) else "")
                for i, claim in enumerate(claims)
            )
            assert rebuilt == text.strip()

    def test_the_claim_that_cited_a_source_is_the_one_judged(self, kb):
        """End to end: written with the marker after the period, the deadline
        claim used to arrive uncited and `strict` deleted a sourced fact."""
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds must be filed within 30 days. [1]",
            verify=verdicts({"verdict": "supported", "rationale": "matches"}),
            verification_mode="strict", k=3,
        )
        assert answer.verification.claims[0].citations == [1]
        assert answer.text == "Refunds must be filed within 30 days. [1]"


class TestExplicitFacets:
    """Retrieval subqueries are not the same thing as answer obligations."""

    def test_explicit_facets_replace_the_subqueries(self, kb):
        seen = {}

        def capture(payload):
            seen["facets"] = payload["facets"]
            return {"claims": ["supported"] * len(payload["claims"]),
                    "facets": [{"facet": f, "covered": True}
                               for f in payload["facets"]]}

        answer = kb.ask_multi(
            "refund deadline",
            subqueries=["refund policy 2024 revision", "refund deadline days"],
            facets=["how long the customer has to ask for a refund"],
            llm=lambda p: "Refunds take 30 days [1].", verify=capture, k=5,
        )
        assert seen["facets"] == [
            "how long the customer has to ask for a refund"
        ], "search-shaped subqueries must not become answer obligations"
        assert answer.verification.complete is True

    def test_subqueries_are_the_default(self, kb):
        seen = {}

        def capture(payload):
            seen["facets"] = payload["facets"]
            return ["supported"] * len(payload["claims"])

        kb.ask_multi("refund deadline",
                     subqueries=["refund deadline", "refund payment"],
                     llm=lambda p: "30 days [1].", verify=capture, k=5)
        # subqueries[0] is the original question, which is not a facet
        assert seen["facets"] == ["refund payment"]

    def test_ask_can_declare_facets_too(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds take 30 days [1].",
            facets=["the deadline", "the payment method"],
            verify=lambda payload: {
                "claims": ["supported"] * len(payload["claims"]),
                "facets": [{"facet": payload["facets"][0], "covered": True},
                           {"facet": payload["facets"][1], "covered": False,
                            "rationale": "not answered"}],
            },
            k=3,
        )
        assert answer.verification.ok is True
        assert answer.verification.complete is False
        assert [f["facet"] for f in answer.verification.uncovered_facets] == [
            "the payment method"
        ]

    def test_empty_facets_disable_the_axis(self, kb):
        answer = kb.ask_multi(
            "refund deadline",
            subqueries=["refund deadline", "refund payment"],
            facets=[],
            llm=lambda p: "30 days [1].",
            verify=verdicts("supported"), k=5,
        )
        assert answer.verification.complete is None, (
            "declaring no obligations is not the same as failing them"
        )


class TestSemanticHardening:
    """The verifier segments and classifies; the library decides nothing about
    meaning. What it must guarantee is that the judgement has everything it
    needs, that every verdict lands, and that a partial result never passes.

    Acceptance criterion: `ok` and `complete` are True only when every claim is
    supported and every facet is fully covered.
    """

    def test_claim_contradicting_the_question_fails_closed(self, kb):
        """The question is a source of facts too. A claim can be false with no
        document involved at all — so the question must reach the verifier."""
        seen = {}

        def judge(payload):
            seen["question"] = payload["question"]
            return [{"verdict": "contradicted",
                     "rationale": "the user said March 1st, the answer says May"}]

        answer = kb.ask(
            "I bought this on March 1st — what is the refund deadline?",
            llm=lambda p: "You bought it on May 1st.",
            verify=judge, verification_mode="repair", k=3,
        )
        assert "March 1st" in seen["question"]
        assert answer.verification.ok is False
        assert answer.verification.claims[0].action == "removed"
        assert "May 1st" not in answer.text

    def test_question_facts_are_not_treated_as_missing(self, kb):
        """A fact the user supplied has no citation and never will. `strict`
        drops uncited claims, but a `question_fact` is not uncited — it is
        sourced from the question, and dropping it would delete a true
        statement for lacking a document that cannot exist."""
        answer = kb.ask(
            "I bought this on March 1st — what is the refund deadline?",
            llm=lambda p: ("You bought it on March 1st. "
                           "Refunds must be filed within 30 days [1]."),
            verify=verdicts(
                {"verdict": "question_fact", "rationale": "stated in the question"},
                {"verdict": "supported", "rationale": "matches [1]"},
            ),
            verification_mode="strict", k=3,
        )
        assert "March 1st" in answer.text
        assert answer.verification.claims[0].action == "kept"
        assert answer.verification.ok is True

    def test_historical_claim_needs_historically_marked_evidence(self, kb):
        """A claim about a superseded rule cited against a current document is
        not supported: differing from today's rule does not prove yesterday's.
        The library's job is to put the deciding metadata in front of the
        judge — status, version and effective_date travel with the evidence."""
        seen = {}

        def judge(payload):
            seen["evidence"] = payload["claims"][0]["evidence"]
            status = seen["evidence"][0]["metadata"].get("status")
            if status != "REVOGADO":
                return [{"verdict": "unsupported",
                         "rationale": f"cited source is {status}, not a "
                                      "historical record of the old rule"}]
            return [{"verdict": "supported", "rationale": "historical source"}]

        answer = kb.ask(
            "what was the old refund deadline?",
            llm=lambda p: "The deadline used to be 90 days [1].",
            verify=judge, verification_mode="repair",
            filters={"status": "VIGENTE"}, k=3,
        )
        assert seen["evidence"][0]["metadata"]["status"] == "VIGENTE"
        assert seen["evidence"][0]["metadata"]["version"] == 2
        assert answer.verification.ok is False
        assert answer.verification.claims[0].verdict == "unsupported"
        assert "90 days" not in answer.text

    def test_composite_facet_needs_every_component(self, kb):
        """A facet covering two things is covered only when both are answered.
        Fidelity does not rescue it: every claim can be supported and the
        answer still be incomplete."""
        def judge(payload):
            return {
                "claims": ["supported"] * len(payload["claims"]),
                "facets": [{"facet": f,
                            "covered": "payment" not in f,
                            "rationale": "the payment half is unanswered"}
                           for f in payload["facets"]],
            }

        answer = kb.ask_multi(
            "how long to request a refund and how am I paid back?",
            subqueries=["refund deadline", "refund deadline and payment method"],
            llm=lambda p: "Refunds must be filed within 30 days [1].",
            verify=judge, k=5,
        )
        report = answer.verification
        assert report.ok is True, "every claim was supported"
        assert report.complete is False, "one facet only half answered"
        assert [f["facet"] for f in report.uncovered_facets] == [
            "refund deadline and payment method"
        ]

    def test_composite_proposition_is_judged_atomically(self, kb):
        """Two propositions inside one sentence are two claims. The verifier
        re-segments; each half gets its own verdict, and repair removes only
        the failing half."""
        def resegmenting(payload):
            assert len(payload["claims"]) == 1, "heuristics see one sentence"
            return [
                {"claim": "Refunds must be filed within 30 days [1]",
                 "verdict": "supported", "rationale": "matches [1]"},
                {"claim": "refunds are paid as store credit [1].",
                 "verdict": "contradicted",
                 "rationale": "[1] says nothing about store credit"},
            ]

        answer = kb.ask(
            "refund deadline and payment",
            llm=lambda p: ("Refunds must be filed within 30 days [1] and "
                           "refunds are paid as store credit [1]."),
            verify=resegmenting, verification_mode="repair", k=3,
        )
        report = answer.verification
        assert report.segmentation == "verifier"
        assert len(report.claims) == 2
        assert report.ok is False
        assert "30 days" in answer.text
        assert "store credit" not in answer.text

    def test_incomplete_return_fails_closed(self, kb):
        """Judging two claims and returning one verdict is not a pass on the
        claim that went unjudged."""
        def half(payload):
            return ["supported"]

        answer = kb.ask_multi(
            "refund deadline and payment",
            subqueries=["refund deadline", "refund payment"],
            llm=lambda p: "Refunds take 30 days [1]. They are instant [1].",
            verify=half, verification_mode="repair", k=5,
        )
        report = answer.verification
        assert "2 claims" in report.error
        assert report.valid is False
        assert report.ok is False
        assert report.complete is False, (
            "the facets were declared and never evaluated"
        )
        assert answer.text == ("Refunds take 30 days [1]. They are instant [1]."), (
            "a broken verifier must not mutate the answer"
        )

    @pytest.mark.parametrize(
        "verdict,covered,expect_ok,expect_complete",
        [
            ("supported", True, True, True),
            ("supported", False, True, False),
            ("unsupported", True, False, True),
            ("unsupported", False, False, False),
        ],
    )
    def test_acceptance_criterion(self, kb, verdict, covered, expect_ok,
                                  expect_complete):
        """`ok` and `complete` are independent axes, and both must hold before
        an answer can be called verified and whole."""
        def judge(payload):
            return {
                "claims": [verdict] * len(payload["claims"]),
                "facets": [{"facet": f, "covered": covered}
                           for f in payload["facets"]],
            }

        answer = kb.ask_multi(
            "refund deadline and payment",
            subqueries=["refund deadline", "refund payment"],
            llm=lambda p: "Refunds take 30 days [1].",
            verify=judge, k=5,
        )
        assert answer.verification.ok is expect_ok
        assert answer.verification.complete is expect_complete


class TestSegmentationStructure:
    """Order, non-overlap and full coverage are structural properties the
    library can check without judging meaning."""

    def test_overlapping_spans_are_rejected(self, kb):
        def overlapping(payload):
            return [
                {"claim": "Refunds take 30 days", "verdict": "supported"},
                {"claim": "30 days [1].", "verdict": "supported"},  # overlaps
            ]

        answer = kb.ask("refund", llm=lambda p: "Refunds take 30 days [1].",
                        verify=overlapping, verification_mode="repair", k=3)
        assert "overlaps or is out of order" in answer.verification.error
        assert answer.text == "Refunds take 30 days [1].", "answer preserved"
        assert answer.verification.ok is False

    def test_out_of_order_spans_are_rejected(self, kb):
        def reversed_order(payload):
            return [
                {"claim": "Two [1].", "verdict": "supported"},
                {"claim": "One [1].", "verdict": "supported"},
            ]

        answer = kb.ask("refund", llm=lambda p: "One [1]. Two [1].",
                        verify=reversed_order, k=3)
        assert answer.verification.error
        assert answer.verification.ok is False

    def test_uncovered_answer_text_is_flagged(self, kb):
        def partial_segmentation(payload):
            return [{"claim": "One [1].", "verdict": "supported"}]

        answer = kb.ask("refund", llm=lambda p: "One [1]. Two [1]. Three [1].",
                        verify=partial_segmentation, k=3)
        issue = answer.verification.structural_issues[0]
        assert "not covered by any claim" in issue
        assert "unverified" in issue

    def test_full_coverage_has_no_structural_issue(self, kb):
        def full(payload):
            return [
                {"claim": "One [1].", "verdict": "supported"},
                {"claim": "Two [1].", "verdict": "supported"},
            ]

        answer = kb.ask("refund", llm=lambda p: "One [1]. Two [1].",
                        verify=full, k=3)
        assert answer.verification.structural_issues == []
        assert answer.verification.valid is True


class TestReplacementMustBeSupported:
    """A replacement is text the verifier wrote, not text the user's model
    produced — it enters the answer only on an explicit endorsement."""

    @pytest.mark.parametrize("verdict", ["uncited", "inference", "question_fact"])
    def test_non_supported_replacement_is_not_accepted(self, kb, verdict):
        answer = kb.ask(
            "refund",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "contradicted", "rationale": "wrong",
                 "replacement": "Some rewritten text [1]."},
                recheck=verdict,
            ),
            verification_mode="repair", allow_replacements=True, k=3,
        )
        assert "Some rewritten text" not in answer.text
        assert answer.verification.claims[0].action == "removed"
        assert answer.verification.claims[0].replacement_verdict == verdict

    def test_supported_replacement_is_accepted(self, kb):
        answer = kb.ask(
            "refund",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "contradicted", "rationale": "wrong",
                 "replacement": "Refunds take 30 days [1]."},
                recheck="supported",
            ),
            verification_mode="repair", allow_replacements=True, k=3,
        )
        assert answer.text == "Refunds take 30 days [1]."


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
