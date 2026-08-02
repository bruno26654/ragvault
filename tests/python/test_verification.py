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
    unchecked — the repair itself can introduce an inaccuracy."""

    def test_rejected_replacement_is_dropped(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "contradicted", "rationale": "wrong",
                 "replacement": "Refunds are also instant, honestly [1]."},
                recheck="unsupported",   # verificador rejeita a própria correção
            ),
            verification_mode="repair", k=3,
        )
        assert "instant" not in answer.text
        claim = answer.verification.claims[0]
        assert claim.replacement_verdict == "unsupported"
        assert claim.action == "removed"
        assert "replacement also rejected" in claim.rationale

    def test_accepted_replacement_is_kept(self, kb):
        answer = kb.ask(
            "refund deadline",
            llm=lambda p: "Refunds are instant [1].",
            verify=verdicts(
                {"verdict": "contradicted", "rationale": "wrong",
                 "replacement": "Refunds must be filed within 30 days [1]."},
                recheck="supported",
            ),
            verification_mode="repair", k=3,
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
               verify=counting, verification_mode="repair", k=3)
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
                        verify=flaky, verification_mode="repair", k=3)
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
