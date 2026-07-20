"""Context assembly: from fused candidates to a model-ready context.

Pipeline: group per document (cap chunks/doc) → dedup by content hash →
MMR-style diversity (token-overlap penalty) → neighbor expansion (previous/
next chunks within the same document and version) → token-budget selection →
ordered formatting → stable citations tied to real stored chunks.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

from .chunking import estimate_tokens


@dataclass
class RetrievedChunk:
    chunk_id: str
    document_id: str
    document_version: int
    chunk_index: int
    text: str
    score: float
    dense_score: Optional[float] = None
    bm25_score: Optional[float] = None
    sparse_score: Optional[float] = None
    title: Optional[str] = None
    uri: Optional[str] = None
    section_path: list[str] = field(default_factory=list)
    page_number: Optional[int] = None
    metadata: dict = field(default_factory=dict)
    token_count: int = 0
    expanded: bool = False
    selection_reason: str = "retrieved"


@dataclass
class Citation:
    index: int
    document_id: str
    document_version: int
    chunk_ids: list[str]
    title: Optional[str] = None
    uri: Optional[str] = None
    section: Optional[str] = None
    page_number: Optional[int] = None
    score: float = 0.0

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "document_id": self.document_id,
            "document_version": self.document_version,
            "chunk_ids": self.chunk_ids,
            "title": self.title,
            "uri": self.uri,
            "section": self.section,
            "page_number": self.page_number,
            "score": self.score,
        }


@dataclass
class RetrievalResult:
    context: str
    chunks: list[RetrievedChunk]
    citations: list[Citation]
    token_count: int
    plan: dict = field(default_factory=dict)
    trace: Optional[dict] = None
    #: True when any selected content was cut to fit the token budget.
    truncated: bool = False

    @property
    def text(self) -> str:
        return self.context

    @property
    def sources(self) -> list[Citation]:
        return self.citations

    @property
    def documents(self) -> list[str]:
        seen: list[str] = []
        for chunk in self.chunks:
            if chunk.document_id not in seen:
                seen.append(chunk.document_id)
        return seen

    def __repr__(self) -> str:
        return (
            f"RetrievalResult(chunks={len(self.chunks)}, "
            f"citations={len(self.citations)}, tokens={self.token_count})"
        )


def _token_set(text: str) -> set[str]:
    return {t for t in text.lower().split() if len(t) > 2}


def _overlap(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def assemble_context(
    candidates: list[RetrievedChunk],
    *,
    token_budget: int,
    max_chunks: int,
    max_chunks_per_document: int,
    mmr_lambda: float,
    fetch_neighbors: Any = None,
    context_window: Optional[dict] = None,
    trace: Optional[dict] = None,
) -> RetrievalResult:
    """Assemble the final context from scored candidates (best-first)."""
    window = context_window or {"before": 0, "after": 0}

    # 1. per-document cap + exact-duplicate removal
    per_doc: dict[str, int] = {}
    seen_texts: set[str] = set()
    capped: list[RetrievedChunk] = []
    dropped_dup = 0
    for chunk in candidates:
        key = " ".join(chunk.text.split())
        if key in seen_texts:
            dropped_dup += 1
            continue
        if per_doc.get(chunk.document_id, 0) >= max_chunks_per_document:
            continue
        seen_texts.add(key)
        per_doc[chunk.document_id] = per_doc.get(chunk.document_id, 0) + 1
        capped.append(chunk)

    # 2. MMR-style diversity: greedy selection penalizing token overlap with
    #    already-selected chunks (works without raw vectors; documented).
    selected: list[RetrievedChunk] = []
    selected_tokens: list[set[str]] = []
    remaining = list(capped)
    while remaining:
        best_i = 0
        best_val = float("-inf")
        for i, chunk in enumerate(remaining):
            redundancy = max(
                (_overlap(_token_set(chunk.text), s) for s in selected_tokens),
                default=0.0,
            )
            value = mmr_lambda * chunk.score - (1 - mmr_lambda) * redundancy
            if value > best_val:
                best_val = value
                best_i = i
        chosen = remaining.pop(best_i)
        selected.append(chosen)
        selected_tokens.append(_token_set(chosen.text))

    # 3. token budget selection (with room for expansion)
    budget_left = token_budget
    final: list[RetrievedChunk] = []
    for chunk in selected:
        if len(final) >= max_chunks:
            break
        tokens = chunk.token_count or estimate_tokens(chunk.text)
        if tokens > budget_left:
            if final:
                chunk.selection_reason = "skipped: over token budget"
                continue
            # Even the best chunk exceeds the budget: truncate it rather
            # than return an empty context.
            keep_chars = max(1, len(chunk.text) * budget_left // tokens)
            chunk.text = chunk.text[:keep_chars].rsplit(" ", 1)[0] or chunk.text[:keep_chars]
            chunk.selection_reason = "truncated to token budget"
            tokens = estimate_tokens(chunk.text)
        chunk.token_count = tokens
        final.append(chunk)
        budget_left -= tokens
        if budget_left <= 0:
            break

    # 4. neighbor expansion (never crosses documents/versions, respects budget)
    if fetch_neighbors and (window.get("before", 0) or window.get("after", 0)):
        expanded: list[RetrievedChunk] = []
        have = {c.chunk_id for c in final}
        for chunk in list(final):
            for neighbor in fetch_neighbors(chunk, window):
                if neighbor.chunk_id in have:
                    continue
                tokens = neighbor.token_count or estimate_tokens(neighbor.text)
                if tokens > budget_left:
                    continue
                neighbor.expanded = True
                neighbor.selection_reason = "expanded neighbor"
                neighbor.token_count = tokens
                have.add(neighbor.chunk_id)
                expanded.append(neighbor)
                budget_left -= tokens
        final.extend(expanded)

    # 5. ordering: group by document (best doc first), chunk order inside
    doc_best: dict[str, float] = {}
    for chunk in final:
        doc_best[chunk.document_id] = max(doc_best.get(chunk.document_id, float("-inf")),
                                          chunk.score)
    final.sort(key=lambda c: (-doc_best[c.document_id], c.document_id, c.chunk_index))

    # 6. merge adjacent runs: chunks from the same document AND version with
    #    consecutive chunk_index (typical after neighbor expansion) become a
    #    single reading-order block. Merging never crosses documents or
    #    versions and never mutates the stored chunk texts (join only).
    runs: list[list[RetrievedChunk]] = []
    for chunk in final:
        last = runs[-1] if runs else None
        if (last
                and last[-1].document_id == chunk.document_id
                and last[-1].document_version == chunk.document_version
                and chunk.chunk_index == last[-1].chunk_index + 1):
            last.append(chunk)
        else:
            runs.append([chunk])

    # 7. citations + formatting (one citation per document, chunk_ids keep
    #    the per-excerpt provenance; merged runs render as one block).
    citations: list[Citation] = []
    doc_to_citation: dict[str, Citation] = {}
    blocks: list[str] = []
    truncated = any("truncated" in c.selection_reason for c in final)
    for run in runs:
        first = run[0]
        citation = doc_to_citation.get(first.document_id)
        if citation is None:
            citation = Citation(
                index=len(citations) + 1,
                document_id=first.document_id,
                document_version=first.document_version,
                chunk_ids=[],
                title=first.title,
                uri=first.uri,
                section=" > ".join(first.section_path) or None,
                page_number=first.page_number,
                score=first.score,
            )
            citations.append(citation)
            doc_to_citation[first.document_id] = citation
        for chunk in run:
            citation.chunk_ids.append(chunk.chunk_id)
            citation.score = max(citation.score, chunk.score)

        header = f"[{citation.index}]"
        if first.title:
            header += f" {first.title}"
        if first.section_path:
            header += f" — {' > '.join(first.section_path)}"
        if first.page_number is not None:
            header += f" (page {first.page_number})"
        body = "\n".join(chunk.text.strip() for chunk in run)
        if truncated and any("truncated" in c.selection_reason for c in run):
            header += " [truncated to fit token budget]"
        blocks.append(f"{header}\n{body}")

    context = "\n\n".join(blocks)
    total_tokens = sum(c.token_count for c in final)
    if trace is not None:
        trace["context"] = {
            "candidates": len(candidates),
            "after_dedup_and_caps": len(capped),
            "duplicates_dropped": dropped_dup,
            "selected": len([c for c in final if not c.expanded]),
            "expanded": len([c for c in final if c.expanded]),
            "token_budget": token_budget,
            "tokens_used": total_tokens,
        }
    if trace is not None:
        trace["context"]["merged_blocks"] = len(runs)
        trace["context"]["truncated"] = truncated
    return RetrievalResult(
        context=context,
        chunks=final,
        citations=citations,
        token_count=total_tokens,
        trace=trace,
        truncated=truncated,
    )
