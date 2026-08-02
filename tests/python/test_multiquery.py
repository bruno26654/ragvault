"""Multi-query pipeline: recall on composed questions, metadata precedence,
distractor containment, citation integrity and safe fallbacks."""

from __future__ import annotations

import time

import pytest

import ragvault


@pytest.fixture
def policy_kb(tmp_path):
    """A corpus with multi-hop evidence, a revoked/current pair and
    semantically similar distractors."""
    kb = ragvault.open(tmp_path / "kb")
    kb.add([
        # multi-hop: the answer needs BOTH of these
        {"id": "refund_window", "text":
         "Refund requests must be filed within 30 days of the purchase date.",
         "metadata": {"status": "VIGENTE", "doc_type": "policy"}},
        {"id": "refund_method", "text":
         "Approved refunds are paid back to the original payment method.",
         "metadata": {"status": "VIGENTE", "doc_type": "policy"}},
        # current vs revoked versions of the same policy group
        {"id": "cancel_v2", "text":
         "Cancellation grants a full refund when requested before shipment.",
         "metadata": {"status": "VIGENTE", "doc_group": "cancellation",
                      "effective_date": "2024-06-01", "version": 2,
                      "doc_type": "policy"}},
        {"id": "cancel_v1", "text":
         "Cancellation grants a partial refund when requested before shipment.",
         "metadata": {"status": "REVOGADO", "doc_group": "cancellation",
                      "effective_date": "2019-01-01", "version": 1,
                      "doc_type": "policy"}},
        # semantically similar but irrelevant distractors
        {"id": "distractor_tax", "text":
         "Tax refunds from the government follow a separate federal schedule.",
         "metadata": {"status": "VIGENTE", "doc_type": "faq"}},
        {"id": "distractor_gift", "text":
         "Gift cards are non-refundable and cannot be cancelled after issue.",
         "metadata": {"status": "VIGENTE", "doc_type": "faq"}},
    ])
    yield kb
    kb.close()


class TestMultiHopRecall:
    def test_composed_question_recalls_all_evidence(self, policy_kb):
        """A single query tends to favour one facet; multi-query must surface
        every document the composed question needs."""
        question = "How long do I have to request a refund and how is it paid back?"
        subs = ["refund request deadline days", "how are approved refunds paid back"]
        result = policy_kb.retrieve_multi(question, subqueries=subs, k=6)
        docs = set(result.documents)
        assert {"refund_window", "refund_method"} <= docs, docs

    def test_subqueries_include_original_question_first(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund rules", subqueries=["refund window", "refund payment"], k=4
        )
        assert result.subqueries[0] == "refund rules"
        assert len(result.subqueries) == 3

    def test_max_subqueries_is_respected(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund rules",
            subqueries=[f"sub {i}" for i in range(20)],
            max_subqueries=3, k=3,
        )
        # original + at most 3 subqueries
        assert len(result.subqueries) == 4

    def test_duplicate_subqueries_are_deduplicated(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund window", subqueries=["refund window", "  Refund   Window  "], k=3
        )
        assert result.subqueries == ["refund window"]

    def test_decompose_callback_is_used(self, policy_kb):
        calls = []

        def decomposer(q):
            calls.append(q)
            return ["refund request deadline days", "how are refunds paid back"]

        result = policy_kb.retrieve_multi(
            "refund deadline and payment method", decompose=decomposer, k=6
        )
        assert calls == ["refund deadline and payment method"]
        assert {"refund_window", "refund_method"} <= set(result.documents)


class TestCoverageGuarantee:
    """RRF alone buries a facet's specialist evidence under documents with
    broad shallow consensus; the per-subquery coverage tier prevents it."""

    def test_each_subquery_top_hit_survives_into_the_context(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund deadline and how it is paid back",
            subqueries=["refund request deadline days",
                        "how are approved refunds paid back"],
            k=6, trace=True,
        )
        reserved = {
            r["chunk_id"] for r in result.trace["fusion"]["coverage_reserved"]
        }
        assert reserved, "coverage must reserve at least one chunk per subquery"
        selected = {c.chunk_id for c in result.chunks}
        assert reserved <= selected, "reserved evidence must reach the context"

    def test_coverage_can_be_disabled(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund deadline", subqueries=["refund payment"],
            coverage_per_subquery=0, k=4, trace=True,
        )
        assert result.trace["fusion"]["coverage_reserved"] == []

    def test_coverage_is_traced_with_fused_scores(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund", subqueries=["refund payment"], k=4, trace=True
        )
        top = result.trace["fusion"]["top"][0]
        assert "fused_score" in top and "coverage_reserved" in top

    def test_single_query_needs_no_coverage_tier(self, policy_kb):
        """With one query there are no facets to protect; scores stay pure RRF."""
        result = policy_kb.retrieve_multi("refund deadline", k=4, trace=True)
        assert result.trace["fusion"]["coverage_reserved"] == []


class TestVersionPrecedence:
    def test_revoked_version_is_dropped_and_reported(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "cancellation refund before shipment", resolve_versions=True, k=6
        )
        assert "cancel_v2" in result.documents
        assert "cancel_v1" not in result.documents, "revoked version must not be cited"
        assert result.conflicts, "the conflict must be represented explicitly"
        conflict = result.conflicts[0]
        assert conflict["kept"]["document_id"] == "cancel_v2"
        assert conflict["dropped"][0]["document_id"] == "cancel_v1"
        assert "REVOGADO" in conflict["dropped"][0]["reason"]

    def test_without_resolve_versions_both_can_appear(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "cancellation refund before shipment", resolve_versions=False, k=6
        )
        assert result.conflicts == []

    def test_mandatory_filter_excludes_revoked_before_search(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "cancellation refund", filters={"status": "VIGENTE"}, k=6
        )
        assert "cancel_v1" not in result.documents
        assert result.plan["filtered"] is True

    def test_effective_date_breaks_tie_when_status_equal(self, tmp_path):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([
                {"id": "new", "text": "Holiday policy allows ten extra days.",
                 "metadata": {"status": "VIGENTE", "doc_group": "holiday",
                              "effective_date": "2025-01-01", "version": 3}},
                {"id": "old", "text": "Holiday policy allows five extra days.",
                 "metadata": {"status": "VIGENTE", "doc_group": "holiday",
                              "effective_date": "2021-01-01", "version": 1}},
            ])
            result = kb.retrieve_multi("holiday extra days", resolve_versions=True, k=5)
            assert "new" in result.documents
            assert "old" not in result.documents
            assert "older" in result.conflicts[0]["dropped"][0]["reason"]

    def test_boost_prefers_document_type(self, policy_kb):
        """Metadata boosts apply after fusion and change the ordering."""
        plain = policy_kb.retrieve_multi("refund", k=6)
        boosted = policy_kb.retrieve_multi(
            "refund", k=6, boosts=[{"filter": {"doc_type": "policy"}, "weight": 5.0}]
        )
        assert boosted.documents[0] != "distractor_tax"
        top_meta = boosted.chunks[0].metadata.get("doc_type")
        assert top_meta == "policy", (top_meta, plain.documents, boosted.documents)


class TestDistractorsAndExpansion:
    def test_distractors_do_not_get_neighbor_expansion(self, tmp_path):
        """Neighbor expansion must only touch chunks that survived the final
        selection — never a distractor document."""
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([
                {"id": "target", "text": " ".join(
                    f"Refund clause {i} explains the reimbursement window."
                    for i in range(12))},
                {"id": "noise", "text": " ".join(
                    f"Unrelated gardening tip {i} about soil pH."
                    for i in range(12))},
            ])
            result = kb.retrieve_multi(
                "refund reimbursement window", k=3,
                context_window={"before": 1, "after": 1},
            )
            expanded_docs = {c.document_id for c in result.chunks if c.expanded}
            assert "noise" not in expanded_docs

    def test_only_retrieved_documents_are_cited(self, policy_kb):
        result = policy_kb.retrieve_multi("refund deadline", k=3)
        cited = {c.document_id for c in result.citations}
        retrieved = set(result.documents)
        assert cited <= retrieved
        # every citation points at real stored chunks
        for citation in result.citations:
            for chunk_id in citation.chunk_ids:
                assert policy_kb._vault.get_chunks([chunk_id])[0] is not None

    def test_global_token_budget_is_respected(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund and cancellation and shipping rules",
            subqueries=["refund", "cancellation", "shipping"],
            k=6, token_budget=60,
        )
        assert result.token_count <= 60


class TestRerankSafety:
    def test_rerank_reorders_without_losing_recall(self, policy_kb):
        def drop_everything_but_one(query, chunks):
            return chunks[:1]  # adversarial: throws away recall

        result = policy_kb.retrieve_multi(
            "refund deadline and payment", k=6,
            subqueries=["refund deadline", "refund payment"],
            rerank=drop_everything_but_one, trace=True,
        )
        # the dropped candidates must be recovered, not lost
        assert result.trace["rerank"]["recovered_dropped"]
        assert len(result.documents) > 1

    def test_failing_reranker_falls_back_to_fused_order(self, policy_kb):
        def exploding(query, chunks):
            raise RuntimeError("reranker unavailable")

        result = policy_kb.retrieve_multi(
            "refund deadline", k=4, rerank=exploding, trace=True
        )
        assert result.chunks, "a failing reranker must not empty the result"
        assert "reranker unavailable" in result.trace["rerank_error"]
        assert result.trace["rerank_fallback"] == "fused order kept"

    def test_rerank_scores_are_traced_before_and_after(self, policy_kb):
        def reverse(query, chunks):
            return list(reversed(chunks))

        result = policy_kb.retrieve_multi(
            "refund", k=4, rerank=reverse, trace=True
        )
        assert result.trace["rerank"]["scores_before"]
        assert result.trace["rerank"]["scores_after"]


class TestDecomposerFallback:
    def test_failing_decomposer_falls_back_to_single_query(self, policy_kb):
        def exploding(question):
            raise RuntimeError("llm down")

        result = policy_kb.retrieve_multi(
            "refund deadline", decompose=exploding, k=4, trace=True
        )
        assert result.subqueries == ["refund deadline"]
        assert "llm down" in result.trace["decompose_error"]
        assert result.chunks, "fallback must still retrieve"

    def test_decomposer_returning_none_is_safe(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund deadline", decompose=lambda q: None, k=4, trace=True
        )
        assert result.subqueries == ["refund deadline"]
        assert result.chunks

    def test_decomposer_returning_garbage_is_filtered(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund deadline", decompose=lambda q: ["", "   ", "refund window"], k=4
        )
        assert result.subqueries == ["refund deadline", "refund window"]


class TestTraceCompleteness:
    def test_trace_has_every_documented_stage(self, policy_kb):
        result = policy_kb.retrieve_multi(
            "refund and cancellation", subqueries=["refund", "cancellation"],
            resolve_versions=True, k=5, trace=True, explain=True,
            boosts=[{"filter": {"doc_type": "policy"}, "weight": 2.0}],
            rerank=lambda q, c: c,
        )
        t = result.trace
        assert t["subqueries"]
        assert t["candidates_per_subquery"]
        assert t["fusion"]["method"] == "weighted_rrf"
        assert t["fusion"]["top"][0]["contributions"]
        assert t["version_conflicts"]
        assert t["boosts"]
        assert t["rerank"]["scores_before"] is not None
        assert "eliminated" in t
        for stage in ("decompose", "embed", "search", "fusion", "assemble", "total"):
            assert stage in t["stage_ms"]

    def test_plan_marks_the_multi_pipeline(self, policy_kb):
        result = policy_kb.retrieve_multi("refund", k=3, explain=True)
        assert result.plan["pipeline"] == "multi_query"
        assert result.plan["fusion"] == "weighted_rrf"


class TestAskMulti:
    def test_answer_keeps_only_real_citations(self, policy_kb):
        def hallucinating_llm(prompt):
            # [1] is real; [9] is invented
            return "Refunds take 30 days [1]. Also unicorns are refundable [9]."

        answer = policy_kb.ask_multi(
            "refund deadline", llm=hallucinating_llm, k=3, trace=True
        )
        assert "[9]" not in answer.text
        assert "[1]" in answer.text

    def test_prompt_forbids_question_facts_as_evidence(self, policy_kb):
        seen = {}

        def capture(prompt):
            seen["prompt"] = prompt
            return "ok"

        policy_kb.ask_multi("refund deadline", llm=capture, k=3)
        assert "not documented evidence" in seen["prompt"]

    def test_conflicts_are_stated_in_the_prompt(self, policy_kb):
        seen = {}

        def capture(prompt):
            seen["prompt"] = prompt
            return "ok"

        policy_kb.ask_multi(
            "cancellation refund", llm=capture, resolve_versions=True, k=6
        )
        assert "Version notes" in seen["prompt"]
        assert "cancel_v1" in seen["prompt"]

    def test_ask_multi_returns_answer_with_provenance(self, policy_kb):
        answer = policy_kb.ask_multi(
            "refund deadline", llm=lambda p: "30 days [1].", k=3
        )
        assert isinstance(answer, ragvault.Answer)
        assert answer.citations
        assert answer.result.subqueries

    def test_bad_llm_is_rejected_actionably(self, policy_kb):
        with pytest.raises(ragvault.ConfigurationError) as err:
            policy_kb.ask_multi("refund", llm="not-callable", k=2)
        assert "callable" in str(err.value)


class TestCompatibility:
    def test_retrieve_is_unchanged(self, policy_kb):
        """The single-query path must behave exactly as before."""
        result = policy_kb.retrieve("refund deadline", k=3)
        assert result.chunks and result.citations
        assert not hasattr(result, "subqueries")

    def test_retrieve_many_still_matches_individual(self, policy_kb):
        queries = ["refund deadline", "cancellation policy"]
        batch = policy_kb.retrieve_many(queries, k=3)
        single = [policy_kb.retrieve(q, k=3) for q in queries]
        for b, s in zip(batch, single):
            assert [c.chunk_id for c in b.chunks] == [c.chunk_id for c in s.chunks]

    def test_ask_is_unchanged(self, policy_kb):
        answer = policy_kb.ask("refund deadline", llm=lambda p: "30 days [1]", k=3)
        assert answer.text == "30 days [1]"

    def test_multi_with_single_query_matches_single_documents(self, policy_kb):
        """With no subqueries the multi pipeline must agree with retrieve()
        on which documents are relevant (fusion of one ranking is order
        preserving)."""
        single = policy_kb.retrieve("refund deadline", k=4)
        multi = policy_kb.retrieve_multi("refund deadline", k=4)
        assert multi.documents[0] == single.documents[0]

    def test_unknown_fusion_is_rejected(self, policy_kb):
        with pytest.raises(ragvault.ConfigurationError):
            policy_kb.retrieve_multi("refund", fusion="magic")

    def test_empty_question_is_rejected(self, policy_kb):
        with pytest.raises(ragvault.ConfigurationError):
            policy_kb.retrieve_multi("   ")

    def test_result_is_a_retrieval_result(self, policy_kb):
        result = policy_kb.retrieve_multi("refund", k=3)
        assert isinstance(result, ragvault.RetrievalResult)
        assert isinstance(result, ragvault.MultiRetrievalResult)


class TestCpuPerformance:
    def test_multi_query_stays_within_a_small_factor_of_single(self, tmp_path):
        """Multi-query runs N searches in one batched native call; on CPU it
        must not cost N times a single query."""
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": f"d{i}", "text": f"policy subject {i % 20} clause {i}"}
                    for i in range(400)])
            kb.flush()
            subs = [f"subject {i}" for i in range(5)]

            for _ in range(3):  # warm up
                kb.retrieve("subject 3", k=5)
                kb.retrieve_multi("subject 3", subqueries=subs, k=5)

            t0 = time.perf_counter()
            for _ in range(10):
                kb.retrieve("subject 3", k=5)
            single = (time.perf_counter() - t0) / 10

            t0 = time.perf_counter()
            for _ in range(10):
                kb.retrieve_multi("subject 3", subqueries=subs, k=5)
            multi = (time.perf_counter() - t0) / 10

        # 6 queries batched must cost well under 6x a single query
        assert multi < single * 6, f"single={single*1000:.2f}ms multi={multi*1000:.2f}ms"
