from __future__ import annotations

import json
import threading
import urllib.request

import pytest

import ragvault


class TestSq8:
    def test_low_memory_preset_uses_sq8_end_to_end(self, tmp_path):
        with ragvault.open(tmp_path / "kb", preset="low_memory") as kb:
            assert kb.config.quantization == "sq8"
            kb.add([{"id": f"d{i}", "text": f"topic number {i} with details"}
                    for i in range(40)])
            result = kb.retrieve("topic number 7", k=3, mode="dense", explain=True)
            assert result.plan["dense_backend"] == "sq8_flat"
            assert result.chunks[0].document_id == "d7"
            stats = kb.stats()
            assert stats["quantization"] == "sq8"
            assert stats["sq8_bytes"] > 0
            assert stats["hnsw_nodes"] == 0, "sq8 mode must not build the graph"
            kb.flush()
        # reopen keeps the quantized backend working
        with ragvault.open(tmp_path / "kb") as kb:
            result = kb.retrieve("topic number 21", k=3, mode="dense", explain=True)
            assert result.plan["dense_backend"] == "sq8_flat"
            assert result.chunks[0].document_id == "d21"

    def test_sq8_matches_unquantized_ranking(self, tmp_path):
        docs = [{"id": f"d{i}", "text": f"unique subject {i} alpha beta"}
                for i in range(60)]
        with ragvault.open(tmp_path / "plain") as plain:
            plain.add(docs)
            expected = [c.document_id
                        for c in plain.retrieve("unique subject 33", k=5,
                                                mode="dense").chunks]
        with ragvault.open(tmp_path / "quant", quantization="sq8") as quant:
            quant.add(docs)
            got = [c.document_id
                   for c in quant.retrieve("unique subject 33", k=5,
                                           mode="dense").chunks]
        assert got[0] == expected[0], "top hit must match the exact backend"


class TestMigrateEmbeddings:
    def test_blocking_migration_swaps_dimension(self, tmp_path):
        kb = ragvault.open(tmp_path / "kb", embedding="builtin:hashed-ngram:256")
        try:
            kb.add([{"id": "a", "text": "cancellation policy text"},
                    {"id": "b", "text": "shipping schedule text"}])
            assert kb.stats()["dim"] == 256
            kb.migrate_embeddings("builtin:hashed-ngram:512")
            assert kb.stats()["dim"] == 512
            assert kb.config.embedding == "builtin:hashed-ngram:512"
            result = kb.retrieve("cancellation policy", k=1)
            assert result.chunks[0].document_id == "a"
        finally:
            kb.close()
        # reopen uses the new embedding space
        with ragvault.open(tmp_path / "kb") as kb:
            assert kb.embedder.dimension == 512
            result = kb.retrieve("shipping schedule", k=1)
            assert result.chunks[0].document_id == "b"

    def test_migration_is_noop_for_same_embedder(self, tmp_path):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add("something")
            generation_before = kb.stats()["generation"]
            kb.migrate_embeddings("builtin:hashed-ngram:512")  # same as default
            assert kb.stats()["generation"] == generation_before

    def test_migration_rejected_on_tenant_view(self, tmp_path):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add("content")
            with pytest.raises(ragvault.ConfigurationError):
                kb.for_tenant("acme").migrate_embeddings("builtin:hashed-ngram:256")

    def test_failed_migration_leaves_old_vault_intact(self, tmp_path):
        import numpy as np

        calls = {"n": 0}

        def flaky_embedder(texts):
            calls["n"] += 1
            if calls["n"] > 1:
                raise RuntimeError("embedder exploded mid-migration")
            return np.zeros((len(texts), 32), dtype=np.float32)

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": f"d{i}", "text": f"document {i}"} for i in range(3)])
            with pytest.raises(Exception):
                kb.migrate_embeddings(flaky_embedder)
            # old vault still fully functional
            assert kb.stats()["dim"] == 512
            assert kb.retrieve("document 1", k=1, mode="keyword").chunks


class TestAdaptersWithoutDeps:
    def test_haystack_and_dspy_errors_are_actionable(self, tmp_path):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add("x")
            for method, needle in [(kb.as_haystack_retriever, "haystack-ai"),
                                   (kb.as_dspy_retriever, "dspy")]:
                try:
                    method()
                except ragvault.ConfigurationError as exc:
                    assert needle in str(exc)
                # if the dep happens to be installed, constructing it is enough


class TestStudio:
    def test_studio_serves_page_stats_and_queries(self, tmp_path):
        from ragvault.studio import serve

        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": "refunds", "text": "Refunds within 30 days."}])
            server = serve(kb, port=0, open_browser=False)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                base = f"http://127.0.0.1:{server.server_port}"
                page = urllib.request.urlopen(f"{base}/").read().decode()
                assert "RagVault Studio" in page
                stats = json.loads(urllib.request.urlopen(f"{base}/api/stats").read())
                assert stats["documents"] == 1
                req = urllib.request.Request(
                    f"{base}/api/query",
                    data=json.dumps({"query": "refund timing", "k": 3}).encode(),
                    headers={"Content-Type": "application/json"},
                )
                data = json.loads(urllib.request.urlopen(req).read())
                assert data["chunks"][0]["document_id"] == "refunds"
                assert "plan" in data and "citations" in data
                # malformed request → clean 400, not a crash
                bad = urllib.request.Request(
                    f"{base}/api/query", data=b"{}",
                    headers={"Content-Type": "application/json"},
                )
                with pytest.raises(urllib.error.HTTPError) as err:
                    urllib.request.urlopen(bad)
                assert err.value.code == 400
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=5)
