"""End-to-end lifecycle: install → open → sync → query → modify → sync →
verify update → remove → sync → verify deletion → flush → close → reopen →
query → evaluate."""

from __future__ import annotations

import json

import pytest

import ragvault


def test_full_lifecycle(tmp_path, docs_dir):
    kb_path = tmp_path / "kb"
    with ragvault.open(kb_path, preset="quality") as kb:
        report = kb.sync(docs_dir)
        assert report.discovered == 3
        assert report.added == 3
        assert report.failed == 0

        # idempotent second sync
        report2 = kb.sync(docs_dir)
        assert report2.added == 0
        assert report2.updated == 0
        assert report2.unchanged == 3

        result = kb.retrieve("how do refunds work when cancelling?", k=3)
        assert "refund" in result.context.lower()
        assert result.citations
        assert result.chunks[0].document_id == "cancellation.md"
        # citations point at real stored chunks
        for citation in result.citations:
            for chunk_id in citation.chunk_ids:
                assert kb._vault.get_chunk(chunk_id) is not None

        # modify a document → sync updates it atomically
        (docs_dir / "shipping.md").write_text(
            "# Shipping\n\nOrders now ship within 2 business days, with express couriers.\n"
        )
        report3 = kb.sync(docs_dir)
        assert report3.updated == 1
        assert report3.unchanged == 2
        result = kb.retrieve("how fast do orders ship?", k=2)
        assert "2 business days" in result.context
        assert "5 business days" not in result.context

        # remove a file → sync deletes it
        (docs_dir / "accounts.txt").unlink()
        report4 = kb.sync(docs_dir)
        assert report4.deleted == 1
        result = kb.retrieve("closing accounts settings page", k=5, mode="keyword")
        assert all(c.document_id != "accounts.txt" for c in result.chunks)

        kb.flush()

    # reopen: everything survives
    with ragvault.open(kb_path) as kb:
        assert {d["document_id"] for d in kb.documents()} == {
            "cancellation.md", "shipping.md",
        }
        result = kb.retrieve("cancellation refunds", k=2)
        assert result.chunks
        assert result.chunks[0].document_id == "cancellation.md"

        # evaluation on a small dataset
        dataset = tmp_path / "eval.jsonl"
        dataset.write_text(
            json.dumps({"query": "refund policy for cancellation",
                        "relevant_ids": ["cancellation.md"]}) + "\n"
            + json.dumps({"query": "shipping speed", "relevant_ids": ["shipping.md"]}) + "\n"
        )
        report = kb.evaluate(dataset, k=3)
        assert report.queries == 2
        assert report.recall_at_k == 1.0
        assert report.mrr == 1.0
        assert report.latency_p95_ms > 0
        assert "recall@3" in report.to_markdown()


def test_add_retrieve_without_sync(kb):
    ids = kb.add(
        ["The moon orbits the earth.", "Postgres is a relational database."],
        metadata={"topic": "misc"},
    )
    assert len(ids) == 2
    result = kb.retrieve("which database is relational?", k=1)
    assert "Postgres" in result.context
    assert result.token_count > 0
    assert result.citations[0].document_id == ids[1]


def test_context_manager_and_reopen_lock(tmp_path):
    path = tmp_path / "kb"
    with ragvault.open(path) as kb:
        kb.add("locked content")
        # second writer must be rejected while open
        with pytest.raises(ragvault.VaultLockedError):
            ragvault.open(path)
    # after close, reopening works
    with ragvault.open(path) as kb:
        assert kb.stats()["documents"] == 1


def test_document_versioning_and_replace(kb):
    kb.add([{"id": "doc-1", "text": "version one content about apples"}])
    kb.add([{"id": "doc-1", "text": "version two content about oranges"}])
    inspection = kb.inspect("doc-1")
    assert inspection["document"]["current_version"] == 2
    assert len(inspection["versions"]) == 2
    result = kb.retrieve("apples", k=3, mode="keyword")
    assert not result.chunks, "old version must be invisible"
    result = kb.retrieve("oranges", k=3, mode="keyword")
    assert result.chunks[0].document_id == "doc-1"


def test_remove_document(kb):
    kb.add([{"id": "gone", "text": "ephemeral content zebra"}])
    assert kb.remove("gone") is True
    assert kb.remove("gone") is False
    result = kb.retrieve("zebra", k=3, mode="keyword")
    assert not result.chunks
