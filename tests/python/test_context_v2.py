"""Adversarial tests for the context builder v2: adjacent-run merging,
truncation flags, budget enforcement after expansion, tenant isolation of
expansion."""

from __future__ import annotations

import ragvault
from ragvault.chunking import estimate_tokens


def test_adjacent_expanded_chunks_merge_into_one_block(tmp_path):
    text = "\n\n".join(f"Paragraph number {i} about topic-{i}." for i in range(30))
    with ragvault.open(tmp_path / "kb") as kb:
        kb.add([{"id": "long", "text": text}])
        if len(kb.inspect("long")["chunks"]) < 3:
            return  # chunking produced too few chunks to exercise merging
        result = kb.retrieve(
            "topic-7", k=1, mode="keyword",
            context_window={"before": 1, "after": 1},
        )
        assert len([c for c in result.chunks if c.expanded]) >= 1
        # retrieved chunk + expanded neighbors are consecutive -> ONE block
        assert result.context.count("[1]") == 1
        assert result.context.count("[2]") == 0, "no spurious second citation"
        # reading order preserved inside the merged block
        indices = [c.chunk_index for c in result.chunks]
        assert indices == sorted(indices) or len(set(
            c.document_id for c in result.chunks)) > 1


def test_merge_never_crosses_documents(tmp_path):
    with ragvault.open(tmp_path / "kb") as kb:
        kb.add([{"id": "a", "text": "alpha shared subject"},
                {"id": "b", "text": "beta shared subject"}])
        result = kb.retrieve("shared subject", k=2)
        assert result.context.count("[1]") == 1
        assert result.context.count("[2]") == 1, "distinct docs keep distinct blocks"


def test_truncation_is_flagged_and_budget_enforced(tmp_path):
    with ragvault.open(tmp_path / "kb") as kb:
        kb.add([{"id": "big", "text": "verylongword " * 800}])
        result = kb.retrieve("verylongword", token_budget=120, k=3)
        assert result.token_count <= 120
        assert result.truncated is True
        assert "[truncated to fit token budget]" in result.context


def test_budget_holds_after_expansion(tmp_path):
    text = "\n\n".join(f"Section paragraph {i} content-{i}." for i in range(40))
    with ragvault.open(tmp_path / "kb") as kb:
        kb.add([{"id": "doc", "text": text}])
        result = kb.retrieve(
            "content-9", k=2, mode="keyword",
            context_window={"before": 2, "after": 2},
            token_budget=250,
        )
        assert result.token_count <= 250, "expansion must respect the final budget"
        # sanity: token_count consistent with the assembled text estimate
        assert result.token_count <= estimate_tokens(result.context) + 50


def test_expansion_respects_tenant_scope(tmp_path):
    with ragvault.open(tmp_path / "kb") as kb:
        acme = kb.for_tenant("acme")
        globex = kb.for_tenant("globex")
        acme.add([{"id": "acme-doc", "text": "acme secret alpha\n\nacme secret beta"}])
        globex.add([{"id": "globex-doc", "text": "globex secret alpha"}])
        result = acme.retrieve("secret alpha", k=5,
                               context_window={"before": 1, "after": 1})
        assert result.chunks
        assert all(c.document_id == "acme-doc" for c in result.chunks), (
            "expansion must never surface another tenant's chunks"
        )


def test_no_truncation_flag_when_budget_fits(tmp_path):
    with ragvault.open(tmp_path / "kb") as kb:
        kb.add("short content fits easily")
        result = kb.retrieve("short content", token_budget=4000)
        assert result.truncated is False
        assert "[truncated" not in result.context
