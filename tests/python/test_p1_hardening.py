"""P1 hardening: raw-bytes source identity, processing fingerprints and
honest preset semantics."""

from __future__ import annotations

import pytest

import ragvault


class TestSourceIdentity:
    def test_distinct_binary_files_are_distinct(self, tmp_path):
        """Two files whose bytes differ only in invalid-UTF8 regions decode
        identically with errors='replace' — identity must come from the raw
        bytes, so sync must treat an in-place change as an update."""
        docs = tmp_path / "docs"
        docs.mkdir()
        # \xff and \xfe both decode to U+FFFD: same decoded text, different bytes.
        (docs / "bin.txt").write_bytes(b"prefix \xff suffix common words here")
        with ragvault.open(tmp_path / "kb") as kb:
            report = kb.sync(docs)
            assert report.added == 1
            (docs / "bin.txt").write_bytes(b"prefix \xfe suffix common words here")
            report = kb.sync(docs)
            assert report.updated == 1, (
                "byte-different file must not be treated as unchanged"
            )

    def test_unchanged_bytes_are_skipped(self, tmp_path, docs_dir):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.sync(docs_dir)
            report = kb.sync(docs_dir)
            assert report.unchanged == report.discovered
            assert report.added == report.updated == 0

    def test_metadata_records_separated_hashes(self, tmp_path):
        docs = tmp_path / "docs"
        docs.mkdir()
        # CSV parsing transforms the content, so the two hashes must differ.
        (docs / "table.csv").write_text("name,role\nAda,engineer\n")
        with ragvault.open(tmp_path / "kb") as kb:
            kb.sync(docs)
            meta = kb.documents()[0]["metadata"]
            assert len(meta["source_content_hash"]) == 64
            assert len(meta["parsed_content_hash"]) == 64
            assert "processing_fingerprint" in meta
            assert meta["source_content_hash"] != meta["parsed_content_hash"], (
                "identity hash (raw bytes) and parsed hash must be distinct "
                "for transforming parsers"
            )

    def test_chunking_change_invalidates(self, tmp_path, docs_dir):
        """Same bytes + different chunking fingerprint => reprocess."""
        path = tmp_path / "kb"
        with ragvault.open(path) as kb:
            kb.sync(docs_dir)
        # target_tokens is an identity-adjacent knob; simulate a pipeline
        # change by editing the stored config (chunking params are fixed at
        # creation via the public API).
        import json
        cfg_path = path / "ragvault.json"
        cfg = json.loads(cfg_path.read_text())
        cfg["target_tokens"] = cfg["target_tokens"] + 50
        cfg_path.write_text(json.dumps(cfg))
        with ragvault.open(path) as kb:
            report = kb.sync(docs_dir)
            assert report.updated == report.discovered, (
                "pipeline change must invalidate previously synced documents"
            )


class TestPresetHonesty:
    def test_quality_without_embedding_errors_actionably(self, tmp_path):
        with pytest.raises(ragvault.ConfigurationError) as err:
            ragvault.open(tmp_path / "kb", preset="quality")
        message = str(err.value)
        assert "sentence-transformers" in message
        assert "offline-lite" in message
        assert "builtin:hashed-ngram" in message

    def test_quality_with_explicit_lexical_fallback_works(self, tmp_path):
        with ragvault.open(tmp_path / "kb", preset="quality",
                           embedding="builtin:hashed-ngram") as kb:
            kb.add("explicit lexical opt-in")
            assert kb.retrieve("lexical opt-in", k=1).chunks

    def test_quality_with_callable_embedder_works(self, tmp_path):
        import numpy as np

        def embedder(texts):
            out = np.zeros((len(texts), 16), dtype=np.float32)
            for i, t in enumerate(texts):
                out[i, hash(t) % 16] = 1.0
            return out

        with ragvault.open(tmp_path / "kb", preset="quality",
                           embedding=embedder) as kb:
            kb.add("semantic-ish content")
            assert kb.stats()["dim"] == 16

    def test_offline_lite_is_explicitly_lexical(self, tmp_path):
        with ragvault.open(tmp_path / "kb", preset="offline-lite") as kb:
            assert kb.config.embedding == "builtin:hashed-ngram"
            kb.add("offline baseline")
            assert kb.retrieve("offline baseline", k=1).chunks

    def test_reopen_of_quality_kb_needs_no_argument(self, tmp_path):
        path = tmp_path / "kb"
        with ragvault.open(path, preset="quality",
                           embedding="builtin:hashed-ngram") as kb:
            kb.add("persisted decision")
        # the embedding decision is stored; reopen is frictionless
        with ragvault.open(path) as kb:
            assert kb.config.preset == "quality"
            assert kb.retrieve("persisted decision", k=1).chunks
