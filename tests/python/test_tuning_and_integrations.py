from __future__ import annotations

import json

import pytest

import ragvault


@pytest.fixture
def eval_kb(tmp_path):
    kb = ragvault.open(tmp_path / "kb")
    kb.add([
        {"id": "refunds", "text": "Refunds are processed within 30 days of cancellation."},
        {"id": "shipping", "text": "Orders ship within five business days nationwide."},
        {"id": "accounts", "text": "Accounts can be closed from the settings page."},
        {"id": "privacy", "text": "Personal data is deleted upon account closure request."},
    ])
    yield kb
    kb.close()


DATASET = [
    {"query": "refund timing after cancellation", "relevant_ids": ["refunds"]},
    {"query": "how fast is shipping", "relevant_ids": ["shipping"]},
    {"query": "close my account", "relevant_ids": ["accounts"]},
]


class TestCompare:
    def test_compare_presets(self, eval_kb):
        report = eval_kb.compare(DATASET, presets=["fast", "quality"], k=3)
        assert set(report.reports) == {"fast", "quality"}
        markdown = report.to_markdown()
        assert "| fast |" in markdown and "| quality |" in markdown
        assert report.best("recall_at_k") in ("fast", "quality")
        # config must be restored after comparison
        assert eval_kb.config.preset == "balanced"

    def test_unknown_preset_rejected(self, eval_kb):
        with pytest.raises(ragvault.ConfigurationError):
            eval_kb.compare(DATASET, presets=["warp-speed"])


class TestTune:
    def test_tune_returns_evidence_and_never_auto_applies(self, eval_kb):
        before = eval_kb.config.ef_search
        rec = eval_kb.tune(
            DATASET, objective="mrr", k=3,
            grid={"ef_search": [48, 96], "bm25_weight": [0.8, 1.2]},
        )
        assert len(rec.trials) == 4
        assert rec.best_score >= max(t["score"] for t in rec.trials) - 1e-9
        assert all("p95_ms" in t for t in rec.trials)
        assert eval_kb.config.ef_search == before, "tune must not change config"
        assert "recommendation" in rec.to_markdown().lower()

    def test_apply_persists_config(self, eval_kb):
        rec = eval_kb.tune(DATASET, objective="mrr", k=3,
                           grid={"candidates": [40, 60]})
        eval_kb.apply(rec)
        assert eval_kb.config.candidates == rec.best_params["candidates"]
        stored = json.loads((eval_kb.path / "ragvault.json").read_text())
        assert stored["candidates"] == rec.best_params["candidates"]

    def test_latency_constraint_impossible_raises(self, eval_kb):
        with pytest.raises(ragvault.EvaluationError):
            eval_kb.tune(DATASET, objective="mrr", k=3,
                         grid={"ef_search": [48]}, max_p95_ms=0.0)

    def test_bad_objective_and_bad_knob(self, eval_kb):
        with pytest.raises(ragvault.ConfigurationError):
            eval_kb.tune(DATASET, objective="vibes", k=3, grid={"ef_search": [48]})
        with pytest.raises(ragvault.ConfigurationError):
            eval_kb.tune(DATASET, k=3, grid={"hnsw_m": [8]})


class TestIntegrations:
    """Real roundtrips against the actual frameworks when installed (the CI
    ``integrations`` job pins real versions and runs these), plus actionable
    errors when a framework is absent."""

    def test_langchain_retriever_roundtrip(self, eval_kb):
        pytest.importorskip("langchain_core")
        retriever = eval_kb.as_langchain_retriever(k=2)
        docs = retriever.invoke("refund timing")
        assert docs
        assert docs[0].metadata["document_id"] == "refunds"
        assert docs[0].metadata["chunk_id"]
        assert "Refunds" in docs[0].page_content

    def test_llamaindex_retriever_roundtrip(self, eval_kb):
        pytest.importorskip("llama_index.core")
        from llama_index.core.schema import QueryBundle

        retriever = eval_kb.as_llamaindex_retriever(k=2)
        nodes = retriever.retrieve(QueryBundle(query_str="refund timing"))
        assert nodes
        assert nodes[0].node.metadata["document_id"] == "refunds"
        assert nodes[0].score is not None
        assert "Refunds" in nodes[0].node.get_content()

    def test_haystack_retriever_roundtrip(self, eval_kb):
        pytest.importorskip("haystack")
        retriever = eval_kb.as_haystack_retriever(k=2)
        out = retriever.run(query="refund timing")
        docs = out["documents"]
        assert docs
        assert docs[0].meta["document_id"] == "refunds"
        assert "Refunds" in docs[0].content

    def test_dspy_retriever_roundtrip(self, eval_kb):
        pytest.importorskip("dspy")
        retriever = eval_kb.as_dspy_retriever(k=2)
        prediction = retriever.forward("refund timing")
        assert prediction.passages
        assert any("Refunds" in p for p in prediction.passages)

    def test_llamaindex_without_dep_gives_actionable_error(self, eval_kb):
        try:
            import llama_index.core  # noqa: F401
            pytest.skip("llama-index-core installed")
        except ImportError:
            pass
        with pytest.raises(ragvault.ConfigurationError) as err:
            eval_kb.as_llamaindex_retriever()
        assert "llama-index-core" in str(err.value)

    def test_haystack_without_dep_gives_actionable_error(self, eval_kb):
        try:
            import haystack  # noqa: F401
            pytest.skip("haystack-ai installed")
        except ImportError:
            pass
        with pytest.raises(ragvault.ConfigurationError) as err:
            eval_kb.as_haystack_retriever()
        assert "haystack-ai" in str(err.value)

    def test_dspy_without_dep_gives_actionable_error(self, eval_kb):
        try:
            import dspy  # noqa: F401
            pytest.skip("dspy installed")
        except ImportError:
            pass
        with pytest.raises(ragvault.ConfigurationError) as err:
            eval_kb.as_dspy_retriever()
        assert "dspy" in str(err.value)


class TestSparsePersistence:
    def test_sparse_search_via_engine_survives_reopen(self, tmp_path):
        """Sparse vectors ride the WAL now: no flush, reopen, still searchable."""
        import numpy as np

        from ragvault import _native

        config = {
            "dim": 4, "metric": "cosine",
            "hnsw": {"m": 16, "ef_construction": 200, "ef_search": 64, "seed": 1},
            "bm25": {"k1": 1.2, "b": 0.75, "lowercase": True},
            "wal_sync": "sync", "flat_threshold": 1000,
        }
        path = str(tmp_path / "vault")
        vault = _native.Vault.open(path, json.dumps(config))
        doc = {"document_id": "d", "current_version": 1, "metadata": {}}
        chunks = [{"chunk_id": "d#0", "document_id": "d", "document_version": 1,
                   "chunk_index": 0, "text": "sparse doc", "metadata": {},
                   "section_path": []}]
        sparse = [{"indices": [2, 7], "values": [1.0, 2.0]}]
        vault.upsert_document(json.dumps(doc), json.dumps(chunks),
                              np.eye(1, 4, dtype=np.float32), json.dumps(sparse))
        vault.close()  # close flushes; reopen and also test compact path

        vault = _native.Vault.open(path, json.dumps(config))
        request = {"k": 5, "mode": "sparse",
                   "sparse": {"indices": [7], "values": [1.0]}}
        hits = vault.search(json.dumps(request))["hits"]
        assert len(hits) == 1 and hits[0]["sparse_score"] == 2.0
        vault.compact()
        hits = vault.search(json.dumps(request))["hits"]
        assert len(hits) == 1, "sparse postings must survive compaction"
        vault.close()


class TestBatchChunkFetch:
    def test_get_chunks_alignment(self, eval_kb):
        ids = [c["chunk_id"] for c in eval_kb.inspect("refunds")["chunks"]]
        got = eval_kb._vault.get_chunks(ids + ["missing-id"])
        assert got[-1] is None
        assert [c["chunk_id"] for c in got[:-1]] == ids
