from __future__ import annotations

import numpy as np
import pytest

import ragvault


class TestIvfPython:
    def test_ivf_flat_end_to_end(self, tmp_path):
        with ragvault.open(tmp_path / "kb", index="ivf_flat", nprobe=64) as kb:
            kb.add([{"id": f"d{i}", "text": f"unique subject {i} details"}
                    for i in range(300)])
            kb.flush()  # trains the IVF
            result = kb.retrieve("unique subject 42", k=3, explain=True)
            assert result.plan["dense_backend"] == "ivf_flat"
            assert result.chunks[0].document_id == "d42"
            assert kb.stats()["ivf"]["nlist"] >= 16
        # reopen retrains from snapshot
        with ragvault.open(tmp_path / "kb") as kb:
            result = kb.retrieve("unique subject 17", k=3, explain=True)
            assert result.plan["dense_backend"] == "ivf_flat"
            assert result.chunks[0].document_id == "d17"

    def test_ivf_below_threshold_falls_back_to_flat(self, tmp_path):
        with ragvault.open(tmp_path / "kb", index="ivf_flat") as kb:
            kb.add([{"id": "a", "text": "small collection"}])
            result = kb.retrieve("small collection", k=1, mode="dense", explain=True)
            assert result.plan["dense_backend"] == "flat"


class TestMmapPython:
    def test_mmap_reopen_matches_memory(self, tmp_path):
        path = tmp_path / "kb"
        with ragvault.open(path) as kb:
            kb.add([{"id": f"d{i}", "text": f"topic {i} content here"}
                    for i in range(30)])
            kb.flush()
            expected = [c.chunk_id for c in kb.retrieve("topic 7 content", k=5).chunks]
        with ragvault.open(path, storage="mmap") as kb:
            assert kb.stats()["storage"] == "mmap"
            got = [c.chunk_id for c in kb.retrieve("topic 7 content", k=5).chunks]
            assert got == expected
            # writes after an mmap open still work (RAM tail)
            kb.add([{"id": "fresh", "text": "fresh tail document"}])
            result = kb.retrieve("fresh tail document", k=1, mode="keyword")
            assert result.chunks[0].document_id == "fresh"
            kb.flush()


class TestFaissCompat:
    def test_roundtrip_export_import(self, tmp_path):
        faiss = pytest.importorskip("faiss")
        from ragvault.compat import faiss as rv_faiss

        with ragvault.open(tmp_path / "src") as kb:
            kb.add([{"id": "a", "text": "cancellation policy"},
                    {"id": "b", "text": "shipping schedule"}])
            index, chunk_ids = rv_faiss.export_to_faiss(kb)
            assert index.ntotal == 2
            # ranking parity: faiss search == kb dense search
            query = kb.embedder.embed_queries(["cancellation policy"])
            _, ids = index.search(np.ascontiguousarray(query), 1)
            faiss_top = chunk_ids[int(ids[0][0])]
            kb_top = kb.retrieve("cancellation policy", k=1, mode="dense").chunks[0]
            assert faiss_top == kb_top.chunk_id

            vectors = rv_faiss.reconstruct_from_faiss(index, chunk_ids)
        with ragvault.open(tmp_path / "dst") as dst:
            rv_faiss.import_vectors(
                dst, ["a", "b"], vectors, texts=["cancellation policy", "shipping schedule"]
            )
            result = dst.retrieve("cancellation policy", k=1, mode="dense")
            assert result.chunks[0].document_id == "a"

    def test_dimension_mismatch_rejected(self, tmp_path):
        pytest.importorskip("faiss")
        from ragvault.compat import faiss as rv_faiss

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add("x")
            with pytest.raises(ragvault.ConfigurationError):
                rv_faiss.import_vectors(kb, ["z"], np.zeros((1, 3), dtype=np.float32))


class _FakeCagraModule:
    """Stand-in for cuvs.neighbors.cagra: brute-force behind the cuVS API
    shape, so the GPU plumbing is testable without hardware."""

    class IndexParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    class SearchParams:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    def build(self, params, vectors):
        return {"vectors": np.asarray(vectors, dtype=np.float32)}

    def search(self, params, index, queries, k):
        data = index["vectors"]
        queries = np.asarray(queries, dtype=np.float32)
        sims = queries @ data.T
        order = np.argsort(-sims, axis=1)[:, :k]
        distances = -np.take_along_axis(sims, order, axis=1)  # cuvs: smaller=closer
        return distances, order


class TestGpuPlumbing:
    def test_sidecar_wiring_with_fake_cuvs(self, tmp_path):
        from ragvault.gpu import CagraDenseSearcher

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": f"d{i}", "text": f"gpu subject {i}",
                     "metadata": {"parity": i % 2}} for i in range(20)])
            searcher = CagraDenseSearcher(kb, cagra_module=_FakeCagraModule())
            assert len(searcher.chunk_ids) == 20

            result = kb.retrieve("gpu subject 7", k=3, dense_searcher=searcher,
                                 explain=True)
            assert result.plan["dense_backend"] == "gpu_sidecar"
            assert result.chunks[0].document_id == "d7"
            assert any("post-filter" in r for r in result.plan["reason"])

            # filters post-applied via the native DSL evaluator
            filtered = kb.retrieve("gpu subject", k=5, dense_searcher=searcher,
                                   filters={"parity": 0}, explain=True)
            assert filtered.chunks
            assert all(int(c.document_id[1:]) % 2 == 0 for c in filtered.chunks)

    def test_failed_searcher_falls_back_to_cpu(self, tmp_path):
        class Exploding:
            def search(self, query, k):
                raise RuntimeError("no device")

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": "a", "text": "fallback content"}])
            result = kb.retrieve("fallback content", k=1,
                                 dense_searcher=Exploding(), explain=True)
            assert result.chunks[0].document_id == "a"
            assert any("fell back to CPU" in r for r in result.plan["reason"])

    def test_gpu_unavailable_error_is_actionable(self, tmp_path):
        try:
            import cuvs  # noqa: F401
            pytest.skip("cuvs installed")
        except ImportError:
            pass
        from ragvault.gpu import CagraDenseSearcher

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add("x")
            with pytest.raises(ragvault.ConfigurationError) as err:
                CagraDenseSearcher(kb)
            assert "cuvs" in str(err.value)

    @pytest.mark.gpu
    def test_real_cuvs_end_to_end(self, tmp_path):
        """Real-hardware validation (see docs/GPU.md runbook):
        pip install cuvs-cu12 && pytest -m gpu"""
        pytest.importorskip("cuvs")
        from ragvault.gpu import CagraDenseSearcher

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": f"d{i}", "text": f"cuda subject {i}"} for i in range(500)])
            searcher = CagraDenseSearcher(kb)
            result = kb.retrieve("cuda subject 42", k=3, dense_searcher=searcher)
            assert result.chunks[0].document_id == "d42"


class TestMaxSim:
    def test_maxsim_score_math(self):
        q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        d = np.array([[1.0, 0.0], [0.7, 0.7]], dtype=np.float32)
        # token1 best match = 1.0; token2 best = 0.7
        assert abs(ragvault.maxsim_score(q, d) - 1.7) < 1e-6
        with pytest.raises(ragvault.EmbeddingError):
            ragvault.maxsim_score(np.zeros((2, 3)), np.zeros((2, 4)))

    def test_maxsim_reranker_end_to_end(self, tmp_path):
        def toy_token_encoder(texts):
            # one "token" per word: 8-dim hash bucket one-hot
            out = []
            for t in texts:
                words = t.lower().split() or [""]
                mat = np.zeros((len(words), 8), dtype=np.float32)
                for i, w in enumerate(words):
                    mat[i, hash(w) % 8] = 1.0
                out.append(mat)
            return out

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": "exact", "text": "cancel refund policy"},
                    {"id": "partial", "text": "cancel other things"}])
            rerank = ragvault.maxsim_reranker(toy_token_encoder)
            result = kb.retrieve("cancel refund policy", k=2, rerank=rerank)
            assert result.chunks[0].document_id == "exact"
            assert result.chunks[0].selection_reason == "maxsim reranked"


class TestDatabase:
    def test_collections_are_isolated(self, tmp_path):
        with ragvault.Database.open(tmp_path / "db") as db:
            docs = db.collection("documents")
            faqs = db.collection("faqs")
            docs.add([{"id": "d", "text": "docs content"}])
            faqs.add([{"id": "f", "text": "faq content"}])
            assert docs.retrieve("content", k=5).documents == ["d"]
            assert faqs.retrieve("content", k=5).documents == ["f"]
            assert db.list_collections() == ["documents", "faqs"]
        # reopen discovers collections on disk
        with ragvault.Database.open(tmp_path / "db") as db:
            assert db.list_collections() == ["documents", "faqs"]

    def test_collection_name_validation(self, tmp_path):
        with ragvault.Database.open(tmp_path / "db") as db:
            for bad in ("../escape", ".hidden", "a/b", "x" * 65, ""):
                with pytest.raises(ragvault.ConfigurationError):
                    db.collection(bad)

    def test_connect_is_explicitly_unimplemented(self):
        with pytest.raises(NotImplementedError) as err:
            ragvault.connect("https://cluster.example.com")
        assert "planned" in str(err.value)
