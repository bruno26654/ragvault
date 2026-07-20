from __future__ import annotations

import threading
import time

import numpy as np
import pytest

import ragvault
from ragvault.chunking import ChunkingConfig, chunk_text
from ragvault.embeddings import HashedNGramEmbedder
from ragvault.parsers import parse_file


class TestChunking:
    def test_markdown_sections_become_section_paths(self):
        text = (
            "# Title\n\nIntro paragraph.\n\n## Section A\n\nContent of A.\n\n"
            "## Section B\n\nContent of B.\n"
        )
        chunks = chunk_text(text, ChunkingConfig(target_tokens=10, max_tokens=20), fmt="markdown")
        paths = {tuple(c.section_path) for c in chunks}
        assert ("Title", "Section A") in paths
        assert ("Title", "Section B") in paths

    def test_oversized_paragraph_is_split(self):
        text = "word " * 3000
        chunks = chunk_text(text, ChunkingConfig(target_tokens=100, max_tokens=150))
        assert len(chunks) > 1
        assert all(c.token_count <= 200 for c in chunks)

    def test_offsets_point_into_source(self):
        text = "First paragraph here.\n\nSecond paragraph there."
        chunks = chunk_text(text, ChunkingConfig(target_tokens=5, max_tokens=10))
        for c in chunks:
            assert text[c.char_start:c.char_end].strip().startswith(c.text.split()[0])

    def test_empty_text(self):
        assert chunk_text("", ChunkingConfig()) == []


class TestParsers:
    def test_markdown_title(self, tmp_path):
        f = tmp_path / "x.md"
        f.write_text("# Hello World\n\nBody.")
        doc = parse_file(f)
        assert doc.title == "Hello World"
        assert doc.format == "markdown"

    def test_html_strips_tags_and_scripts(self, tmp_path):
        f = tmp_path / "x.html"
        f.write_text(
            "<html><head><title>Page Title</title><script>evil()</script></head>"
            "<body><p>Visible text.</p></body></html>"
        )
        doc = parse_file(f)
        assert doc.title == "Page Title"
        assert "Visible text." in doc.text
        assert "evil" not in doc.text

    def test_csv_rows(self, tmp_path):
        f = tmp_path / "x.csv"
        f.write_text("name,role\nAda,engineer\nGrace,admiral\n")
        doc = parse_file(f)
        assert "name: Ada" in doc.text
        assert "role: admiral" in doc.text

    def test_jsonl_text_field(self, tmp_path):
        f = tmp_path / "x.jsonl"
        f.write_text('{"text": "line one"}\n{"text": "line two"}\n')
        doc = parse_file(f)
        assert "line one" in doc.text

    def test_invalid_json_raises_ingestion_error(self, tmp_path):
        f = tmp_path / "x.json"
        f.write_text("{not json")
        with pytest.raises(ragvault.IngestionError) as err:
            parse_file(f)
        assert "json" in str(err.value)

    def test_pdf_without_extra_gives_actionable_error(self, tmp_path):
        try:
            import pypdf  # noqa: F401
            pytest.skip("pypdf installed")
        except ImportError:
            pass
        f = tmp_path / "x.pdf"
        f.write_bytes(b"%PDF-1.4 fake")
        with pytest.raises(ragvault.IngestionError) as err:
            parse_file(f)
        assert "ragvault[pdf]" in str(err.value)


class TestEmbeddings:
    def test_deterministic_and_normalized(self):
        emb = HashedNGramEmbedder(dimension=128)
        a = emb.embed_documents(["hello world"])
        b = emb.embed_documents(["hello world"])
        np.testing.assert_array_equal(a, b)
        assert abs(np.linalg.norm(a[0]) - 1.0) < 1e-5

    def test_similar_texts_are_closer(self):
        emb = HashedNGramEmbedder(dimension=256)
        vecs = emb.embed_documents([
            "cancellation and refund policy",
            "policy for refunds after cancellation",
            "gpu kernels for matrix multiplication",
        ])
        sim_related = float(vecs[0] @ vecs[1])
        sim_unrelated = float(vecs[0] @ vecs[2])
        assert sim_related > sim_unrelated

    def test_empty_text_is_zero_vector(self):
        emb = HashedNGramEmbedder(dimension=64)
        v = emb.embed_documents([""])
        assert np.all(v == 0)


class TestEmbeddingCache:
    def test_cache_hits_on_resync(self, tmp_path, docs_dir):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.sync(docs_dir)
            first_misses = kb.stats()["embedding_cache"]["misses"]
            assert first_misses > 0
            # touch mtime but keep content — hashes unchanged, cache irrelevant
            kb.sync(docs_dir)
            assert kb.stats()["embedding_cache"]["misses"] == first_misses


class TestConcurrency:
    def test_gil_released_during_search(self, tmp_path):
        """A CPU-heavy native search must not block other Python threads."""
        with ragvault.open(tmp_path / "kb", flat_threshold=10**9) as kb:
            kb.add([{"id": f"d{i}", "text": f"document number {i} " + "filler " * 30}
                    for i in range(300)])
            ticks = []
            stop = threading.Event()

            def ticker():
                while not stop.is_set():
                    ticks.append(time.monotonic())
                    time.sleep(0.001)

            t = threading.Thread(target=ticker)
            t.start()
            try:
                for _ in range(30):
                    kb.retrieve("document number filler", k=10, candidates=200)
            finally:
                stop.set()
                t.join()
            assert len(ticks) > 20, "ticker thread must progress during native search"

    def test_concurrent_reads_are_safe(self, tmp_path):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": f"d{i}", "text": f"topic {i} content"} for i in range(50)])
            errors = []

            def worker():
                try:
                    for _ in range(20):
                        kb.retrieve("topic content", k=5)
                except Exception as exc:  # pragma: no cover
                    errors.append(exc)

            threads = [threading.Thread(target=worker) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert not errors


class TestCompaction:
    def test_compact_preserves_results(self, tmp_path):
        with ragvault.open(tmp_path / "kb") as kb:
            kb.add([{"id": f"d{i}", "text": f"unique topic {i}"} for i in range(30)])
            for i in range(15):
                kb.remove(f"d{i}")
            before = kb.retrieve("unique topic 20", k=3, mode="keyword")
            kb.compact()
            after = kb.retrieve("unique topic 20", k=3, mode="keyword")
            assert [c.chunk_id for c in before.chunks] == [c.chunk_id for c in after.chunks]
            assert kb.stats()["tombstones"] == 0


class TestConfig:
    def test_presets_exist_and_resolve(self):
        for preset in ragvault.PRESETS:
            config = ragvault.Config.resolve(preset=preset)
            assert config.preset == preset

    def test_unknown_preset_rejected(self, tmp_path):
        with pytest.raises(ragvault.ConfigurationError):
            ragvault.open(tmp_path / "kb", preset="turbo")

    def test_explain_and_export(self, tmp_path):
        with ragvault.open(tmp_path / "kb", preset="quality",
                           embedding="builtin:hashed-ngram") as kb:
            text = kb.config.explain()
            assert "preset: quality" in text
            out = tmp_path / "config.json"
            kb.config.export(out)
            assert out.exists()
