"""Multi-query retrieval pipeline: ``kb.retrieve_multi()`` / ``kb.ask_multi()``.

Built for composed questions where a single query loses recall. Stages:

1. **Decomposition** — subqueries from an optional external LLM callback or
   supplied manually. The original question is always kept as the anchor
   query, and any decomposition failure falls back safely to single-query.
2. **Batch search** — all queries go through the native parallel
   ``search_many`` (one embedding batch, GIL released).
3. **Global fusion** — Weighted RRF (k0=60) across the per-query rankings,
   deduplicating by ``chunk_id``. Deterministic: ties break by chunk_id.
   RRF alone is *not* enough for composed questions: with k0=60 the gap
   between rank 1 and rank 10 is only ~16%, so a mediocre document ranking
   mid-pack for *every* subquery accumulates more mass than the specialist
   document that is the top hit for exactly one facet — and the facet's
   evidence is silently dropped. A **per-subquery coverage guarantee**
   (``coverage_per_subquery``, default 1) therefore reserves the top hits of
   each subquery into a priority tier, ordered by fused score inside the
   tier. This is what makes multi-hop recall actually improve; measured in
   benchmarks/RESULTS-MULTIQUERY.md.
4. **Version resolution** — optional metadata precedence (status such as
   VIGENTE/REVOGADO, effective date, numeric version) within groups sharing a
   ``doc_group`` metadata key. Losing versions are eliminated *explicitly*:
   they appear in ``result.conflicts`` and in the trace, never silently.
5. **Metadata boosts** — post-fusion multiplicative boosts via native filter
   evaluation. (Mandatory filters run *before* search as native prefilters.)
6. **Global rerank** — optional, recall-safe: the reranker reorders a bounded
   window and can never drop candidates; on failure the fused order stands.
7. **Context assembly** — the existing MMR + dedup + token-budget +
   adjacent-merge builder. Neighbor expansion happens only for chunks that
   survived final selection, so distractors never get expanded.

Provenance is preserved end to end: citations always point at real stored
chunks of retrieved documents, and ``ask_multi`` strips citation markers the
context does not actually contain.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Optional, Sequence

import numpy as np

from .context import Citation, RetrievalResult, RetrievedChunk, assemble_context
from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from .kb import Answer, KnowledgeBase

RRF_K0 = 60.0

#: Default status precedence for version resolution (case-insensitive).
#: Earlier = higher precedence. Statuses not listed rank between the two ends.
ACTIVE_STATUSES = ("vigente", "active", "current", "in_force")
REVOKED_STATUSES = ("revogado", "revoked", "superseded", "obsolete", "expired")


@dataclass
class MultiRetrievalResult(RetrievalResult):
    """RetrievalResult plus multi-query provenance."""

    #: The queries actually executed (original question first).
    subqueries: list[str] = field(default_factory=list)
    #: Version conflicts resolved by metadata precedence: for each group,
    #: which document won and which were eliminated (and why).
    conflicts: list[dict] = field(default_factory=list)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


def _status_rank(status: Optional[str]) -> int:
    """Lower is better. Active < unknown < revoked."""
    if status is None:
        return 1
    s = str(status).strip().lower()
    if s in ACTIVE_STATUSES:
        return 0
    if s in REVOKED_STATUSES:
        return 2
    return 1


def _build_queries(
    question: str,
    subqueries: Optional[Sequence[str]],
    decompose: Optional[Callable[[str], Sequence[str]]],
    max_subqueries: int,
    trace: Optional[dict],
) -> list[str]:
    """Original question first, then validated subqueries. Any decomposition
    problem degrades safely to the single original question."""
    subs: list[str] = []
    error: Optional[str] = None
    if subqueries is not None:
        subs = [s for s in subqueries if isinstance(s, str) and s.strip()]
    elif decompose is not None:
        try:
            raw = decompose(question)
            if raw is None:
                raise ValueError("decompose returned None")
            subs = [s for s in raw if isinstance(s, str) and s.strip()]
        except Exception as exc:  # safe fallback, recorded honestly
            error = f"{type(exc).__name__}: {exc}"
            subs = []

    queries = [question]
    seen = {_normalize(question)}
    for s in subs:
        key = _normalize(s)
        if key in seen:
            continue
        seen.add(key)
        queries.append(s.strip())
        if len(queries) - 1 >= max_subqueries:
            break
    if trace is not None:
        trace["subqueries"] = list(queries)
        if error is not None:
            trace["decompose_error"] = error
            trace["decompose_fallback"] = "single-query (original question only)"
    return queries


def retrieve_multi(
    kb: "KnowledgeBase",
    question: str,
    *,
    subqueries: Optional[Sequence[str]] = None,
    decompose: Optional[Callable[[str], Sequence[str]]] = None,
    max_subqueries: int = 6,
    k: int = 8,
    fusion: str = "weighted_rrf",
    fusion_weights: Optional[Sequence[float]] = None,
    coverage_per_subquery: int = 1,
    rerank: Optional[Callable] = None,
    rerank_window: int = 32,
    filters: Optional[dict] = None,
    boosts: Optional[Sequence[dict]] = None,
    resolve_versions: bool = False,
    version_group_field: str = "doc_group",
    status_field: str = "status",
    effective_date_field: str = "effective_date",
    version_field: str = "version",
    token_budget: Optional[int] = None,
    candidates: Optional[int] = None,
    mode: Optional[str] = None,
    context_window: Optional[dict] = None,
    max_chunks_per_document: Optional[int] = None,
    explain: bool = False,
    trace: bool = False,
) -> MultiRetrievalResult:
    """Multi-query retrieval with global fusion. See module docstring."""
    from . import _native

    if not question or not question.strip():
        raise ConfigurationError("question must be a non-empty string")
    if fusion != "weighted_rrf":
        raise ConfigurationError(
            f"unknown fusion {fusion!r}; available: 'weighted_rrf'"
        )
    if max_subqueries < 0:
        raise ConfigurationError("max_subqueries must be >= 0")

    trace_data: Optional[dict] = {} if trace else None
    stage_ms: dict[str, float] = {}
    t_total = time.monotonic()

    # -- 1. queries ---------------------------------------------------------
    t0 = time.monotonic()
    queries = _build_queries(question, subqueries, decompose, max_subqueries, trace_data)
    stage_ms["decompose"] = (time.monotonic() - t0) * 1000

    weights = list(fusion_weights) if fusion_weights is not None else [1.0] * len(queries)
    if len(weights) != len(queries):
        raise ConfigurationError(
            f"fusion_weights has {len(weights)} entries for {len(queries)} queries "
            "(original question first, then subqueries)"
        )

    # -- 2. batch search (native, mandatory filters as prefilter) -----------
    search_mode = mode or kb.config.retrieval_mode
    pool = candidates or kb.config.candidates
    token_budget = token_budget or kb.config.default_token_budget
    merged_filter = kb._merged_filter(filters)
    if merged_filter is not None:
        _native.validate_filter(json.dumps(merged_filter))

    t0 = time.monotonic()
    vectors = None
    if search_mode in ("dense", "hybrid", "auto"):
        vectors = np.ascontiguousarray(
            kb.embedder.embed_queries(queries), dtype=np.float32
        )
    stage_ms["embed"] = (time.monotonic() - t0) * 1000

    t0 = time.monotonic()
    requests = [{
        "text": q if search_mode in ("keyword", "hybrid", "auto") else None,
        "k": max(pool, k),
        "mode": search_mode,
        "candidates": pool,
        "filter": merged_filter,
        "ef_search": kb.config.ef_search,
        "nprobe": kb.config.nprobe,
        "weights": {
            "dense": kb.config.dense_weight,
            "bm25": kb.config.bm25_weight,
            "sparse": kb.config.sparse_weight,
        },
    } for q in queries]
    responses = kb._vault.search_many(json.dumps(requests), vectors)
    stage_ms["search"] = (time.monotonic() - t0) * 1000

    if trace_data is not None:
        trace_data["candidates_per_subquery"] = {
            q: [{"chunk_id": h["chunk_id"], "score": h["score"]}
                for h in resp["hits"][:pool]]
            for q, resp in zip(queries, responses)
        }

    # -- 3. weighted-RRF fusion + chunk_id dedup ----------------------------
    t0 = time.monotonic()
    fused: dict[str, float] = {}
    contributions: dict[str, dict[str, float]] = {}
    best_hit: dict[str, dict] = {}
    for q, w, resp in zip(queries, weights, responses):
        for rank, hit in enumerate(resp["hits"]):
            cid = hit["chunk_id"]
            part = w / (RRF_K0 + rank + 1.0)
            fused[cid] = fused.get(cid, 0.0) + part
            contributions.setdefault(cid, {})[q] = part
            if cid not in best_hit or hit["score"] > best_hit[cid]["score"]:
                best_hit[cid] = hit
    # Per-subquery coverage guarantee: reserve the top hits of each subquery
    # into a priority tier so a facet's specialist evidence cannot be buried
    # by a mediocre document with broad, shallow consensus (see module doc).
    reserved: dict[str, str] = {}  # chunk_id -> which subquery reserved it
    if coverage_per_subquery > 0 and len(queries) > 1:
        for q, resp in zip(queries, responses):
            taken = 0
            for hit in resp["hits"]:
                if taken >= coverage_per_subquery:
                    break
                cid = hit["chunk_id"]
                if cid in reserved:
                    continue  # already covers another subquery
                reserved[cid] = q
                taken += 1

    tier_lift = (max(fused.values()) if fused else 0.0) + 1.0
    ranked = {
        cid: score + (tier_lift if cid in reserved else 0.0)
        for cid, score in fused.items()
    }
    ordered = sorted(ranked.items(), key=lambda kv: (-kv[1], kv[0]))
    stage_ms["fusion"] = (time.monotonic() - t0) * 1000
    if trace_data is not None:
        trace_data["fusion"] = {
            "method": "weighted_rrf",
            "k0": RRF_K0,
            "unique_chunks": len(ordered),
            "coverage_per_subquery": coverage_per_subquery,
            "coverage_reserved": [
                {"chunk_id": cid, "for_subquery": q} for cid, q in reserved.items()
            ],
            "top": [
                {"chunk_id": cid, "score": ranked[cid], "fused_score": fused[cid],
                 "coverage_reserved": cid in reserved,
                 "contributions": contributions[cid]}
                for cid, _ in ordered[:50]
            ],
        }

    # -- 4. materialize chunks (provenance from real stored chunks) ---------
    eliminated: list[dict] = []
    keep = ordered[: max(k * 4, rerank_window)]
    doc_cache: dict[str, Optional[dict]] = {}
    stored = kb._vault.get_chunks([cid for cid, _ in keep])
    chunks: list[RetrievedChunk] = []
    for (cid, score), raw in zip(keep, stored):
        if raw is None:
            eliminated.append({"chunk_id": cid, "reason": "chunk no longer stored"})
            continue
        hit = dict(best_hit[cid])
        hit["score"] = score
        doc_id = hit["document_id"]
        if doc_id not in doc_cache:
            doc_cache[doc_id] = kb._vault.get_document(doc_id)
        chunks.append(kb._hit_to_chunk(hit, raw, doc_cache[doc_id]))

    # -- 5. version resolution by metadata precedence -----------------------
    conflicts: list[dict] = []
    if resolve_versions:
        t0 = time.monotonic()
        groups: dict[str, dict[str, dict]] = {}
        for chunk in chunks:
            group = chunk.metadata.get(version_group_field)
            if group is None:
                continue
            docs = groups.setdefault(str(group), {})
            entry = docs.setdefault(chunk.document_id, {
                "document_id": chunk.document_id,
                "status": chunk.metadata.get(status_field),
                "effective_date": chunk.metadata.get(effective_date_field)
                or chunk.metadata.get("valid_from"),
                "version": chunk.metadata.get(version_field),
                "best_score": chunk.score,
            })
            entry["best_score"] = max(entry["best_score"], chunk.score)

        losers: dict[str, str] = {}  # document_id -> reason
        for group, docs in groups.items():
            if len(docs) < 2:
                continue

            def precedence(e: dict):
                version = e["version"]
                try:
                    version_num = float(version)
                except (TypeError, ValueError):
                    version_num = float("-inf")
                return (
                    _status_rank(e["status"]),          # active first
                    -(_iso_key(e["effective_date"])),   # latest date first
                    -version_num,                       # highest version first
                    e["document_id"],                   # deterministic tie-break
                )

            ranked = sorted(docs.values(), key=precedence)
            winner, rest = ranked[0], ranked[1:]
            dropped = []
            for e in rest:
                reason = _precedence_reason(winner, e, status_field)
                losers[e["document_id"]] = reason
                dropped.append({"document_id": e["document_id"],
                                "status": e["status"], "reason": reason})
            conflicts.append({
                "group": group,
                "kept": {"document_id": winner["document_id"],
                         "status": winner["status"],
                         "effective_date": winner["effective_date"],
                         "version": winner["version"]},
                "dropped": dropped,
            })

        if losers:
            surviving = []
            for chunk in chunks:
                if chunk.document_id in losers:
                    eliminated.append({
                        "chunk_id": chunk.chunk_id,
                        "document_id": chunk.document_id,
                        "reason": f"version resolution: {losers[chunk.document_id]}",
                    })
                else:
                    surviving.append(chunk)
            chunks = surviving
        stage_ms["resolve_versions"] = (time.monotonic() - t0) * 1000
        if trace_data is not None:
            trace_data["version_conflicts"] = conflicts

    # -- 6. metadata boosts (post-fusion, multiplicative) --------------------
    if boosts:
        t0 = time.monotonic()
        applied = []
        ids = [c.chunk_id for c in chunks]
        for boost in boosts:
            bfilter = boost.get("filter")
            weight = float(boost.get("weight", 1.0))
            if bfilter is None:
                raise ConfigurationError("each boost needs a 'filter' dict")
            _native.validate_filter(json.dumps(bfilter))
            if not ids or weight == 1.0:
                continue
            mask = kb._vault.filter_chunks(ids, json.dumps(bfilter))
            hits = 0
            for chunk, ok in zip(chunks, mask):
                if ok:
                    chunk.score *= weight
                    hits += 1
            applied.append({"filter": bfilter, "weight": weight, "matched": hits})
        chunks.sort(key=lambda c: (-c.score, c.chunk_id))
        stage_ms["boosts"] = (time.monotonic() - t0) * 1000
        if trace_data is not None:
            trace_data["boosts"] = applied

    # -- 7. global rerank (recall-safe, tolerant) ----------------------------
    if rerank is not None:
        t0 = time.monotonic()
        window = min(max(rerank_window, k), len(chunks))
        before_scores = {c.chunk_id: c.score for c in chunks[:window]}
        try:
            reordered = list(rerank(question, list(chunks[:window])))
            # Recall safety: the reranker may only reorder — anything it
            # dropped is appended back in fused order, and the tail beyond
            # the window is untouched.
            seen_ids = {c.chunk_id for c in reordered}
            recovered = [c for c in chunks[:window] if c.chunk_id not in seen_ids]
            chunks = reordered + recovered + chunks[window:]
            if trace_data is not None:
                trace_data["rerank"] = {
                    "window": window,
                    "scores_before": before_scores,
                    "scores_after": {c.chunk_id: c.score for c in chunks[:window]},
                    "recovered_dropped": [c.chunk_id for c in recovered],
                }
        except Exception as exc:
            if trace_data is not None:
                trace_data["rerank_error"] = f"{type(exc).__name__}: {exc}"
                trace_data["rerank_fallback"] = "fused order kept"
        stage_ms["rerank"] = (time.monotonic() - t0) * 1000

    if trace_data is not None and eliminated:
        trace_data["eliminated"] = eliminated

    # -- 8. MMR + context assembly (global budget; expansion only for the
    #       final selection, so distractors never get expanded) --------------
    t0 = time.monotonic()

    def fetch_neighbors(chunk: RetrievedChunk, window: dict) -> list[RetrievedChunk]:
        out: list[RetrievedChunk] = []
        doc_chunks = kb._vault.get_document_chunks(chunk.document_id)
        by_index = {c["chunk_index"]: c for c in doc_chunks}
        for delta in range(-window.get("before", 0), window.get("after", 0) + 1):
            if delta == 0:
                continue
            neighbor = by_index.get(chunk.chunk_index + delta)
            if neighbor and neighbor.get("document_version") == chunk.document_version:
                out.append(kb._hit_to_chunk(
                    {"chunk_id": neighbor["chunk_id"],
                     "document_id": chunk.document_id, "score": 0.0},
                    neighbor,
                    kb._vault.get_document(chunk.document_id),
                ))
        return out

    base = assemble_context(
        chunks[: k * 3],
        token_budget=token_budget,
        max_chunks=k,
        max_chunks_per_document=(
            max_chunks_per_document or kb.config.max_chunks_per_document
        ),
        mmr_lambda=kb.config.mmr_lambda,
        fetch_neighbors=fetch_neighbors,
        context_window=context_window or kb.config.context_window,
        trace=trace_data,
    )
    stage_ms["assemble"] = (time.monotonic() - t0) * 1000
    stage_ms["total"] = (time.monotonic() - t_total) * 1000

    plan = {
        "pipeline": "multi_query",
        "subqueries": len(queries),
        "fusion": "weighted_rrf",
        "candidate_pool_per_query": pool,
        "mode": search_mode,
        "filtered": merged_filter is not None,
        "resolve_versions": resolve_versions,
        "reranked": rerank is not None,
    }
    if responses:
        plan["dense_backend"] = responses[0].get("plan", {}).get("dense_backend")
    if explain or trace:
        plan["token_budget"] = token_budget
    if trace_data is not None:
        trace_data["stage_ms"] = {s: round(ms, 3) for s, ms in stage_ms.items()}

    return MultiRetrievalResult(
        context=base.context,
        chunks=base.chunks,
        citations=base.citations,
        token_count=base.token_count,
        plan=plan,
        trace=trace_data,
        truncated=base.truncated,
        subqueries=queries,
        conflicts=conflicts,
    )


def _iso_key(date: Optional[str]) -> float:
    """Sortable key for ISO-like date strings; unknown dates sort lowest."""
    if not date:
        return float("-inf")
    digits = re.sub(r"[^0-9]", "", str(date))[:14]
    if not digits:
        return float("-inf")
    return float(digits.ljust(14, "0"))


def _precedence_reason(winner: dict, loser: dict, status_field: str) -> str:
    if _status_rank(loser["status"]) > _status_rank(winner["status"]):
        return (f"{status_field}={loser['status']!r} superseded by "
                f"{status_field}={winner['status']!r} ({winner['document_id']})")
    if _iso_key(loser["effective_date"]) < _iso_key(winner["effective_date"]):
        return (f"effective_date {loser['effective_date']!r} older than "
                f"{winner['effective_date']!r} ({winner['document_id']})")
    return f"lower precedence than {winner['document_id']}"


_CITATION_RE = re.compile(r"\[(\d+)\]")


def sanitize_citations(text: str, valid_count: int) -> tuple[str, list[int]]:
    """Remove ``[n]`` markers that do not correspond to a real citation in the
    provided context. Returns (clean_text, removed_indices)."""
    removed: list[int] = []

    def _sub(match: "re.Match[str]") -> str:
        n = int(match.group(1))
        if 1 <= n <= valid_count:
            return match.group(0)
        removed.append(n)
        return ""

    clean = _CITATION_RE.sub(_sub, text)
    if removed:
        clean = re.sub(r"[ \t]{2,}", " ", clean)
    return clean, removed


def ask_multi(
    kb: "KnowledgeBase",
    question: str,
    *,
    llm: Callable[[str], str],
    citations: bool = True,
    system_prompt: Optional[str] = None,
    verify: Optional[Callable] = None,
    verification_mode: str = "report",
    **retrieve_kwargs: Any,
) -> "Answer":
    """Multi-query ask: retrieve_multi + user-provided LLM + citation
    integrity. Markers ``[n]`` that do not exist in the context are stripped
    from the answer, and the prompt forbids presenting facts from the
    question itself as documented evidence.

    Pass ``verify=`` to additionally run post-generation semantic validation:
    citation-marker sanity catches *invented* numbers, while verification
    catches a marker that exists but does not actually support its claim
    (see :mod:`ragvault.verification`).
    """
    from .kb import Answer
    from .verification import verify_answer

    result = retrieve_multi(kb, question, **retrieve_kwargs)
    instructions = system_prompt or (
        "Answer the question using only the context below. "
        + ("Cite sources with [n] markers that match the numbered context "
           "blocks; never invent citation numbers. "
           if citations else "")
        + "Facts stated in the question are not documented evidence — only "
        "the context is. If the context does not contain the answer, say so."
    )
    conflict_note = ""
    if result.conflicts:
        lines = []
        for c in result.conflicts:
            dropped = ", ".join(
                f"{d['document_id']} ({d.get('status')})" for d in c["dropped"]
            )
            lines.append(
                f"- group {c['group']!r}: using {c['kept']['document_id']} "
                f"({c['kept'].get('status')}); superseded: {dropped}"
            )
        conflict_note = (
            "\n\n# Version notes\nConflicting document versions were resolved "
            "by metadata precedence:\n" + "\n".join(lines)
        )
    # Facet checklist: the decomposition already guarantees coverage in
    # *retrieval*; without this it does nothing for coverage in the *answer*,
    # so a model handed both facts can still answer only one facet. Listing
    # the facets closes that gap — but note the explicit escape hatch: a facet
    # with no evidence must be declared unanswered, never invented. Without it
    # "do not omit any" would push the model to fabricate exactly where
    # retrieval came up empty.
    facets = [q for q in result.subqueries[1:] if q.strip()]
    coverage_note = ""
    if facets:
        coverage_note = (
            "\n\n# Required answer facets\n"
            "The question was decomposed into the facets below. Address each "
            "one that the context supports, and do not silently omit any. If "
            "the context has no evidence for a facet, say so explicitly for "
            "that facet instead of guessing.\n"
            + "\n".join(f"- {q}" for q in facets)
        )
        if result.trace is not None:
            result.trace["answer_facets"] = list(facets)

    prompt = (
        f"{instructions}\n\n# Context\n{result.context}{conflict_note}"
        f"{coverage_note}\n\n"
        f"# Question\n{question}\n\n# Answer\n"
    )
    if callable(llm):
        text = llm(prompt)
    elif hasattr(llm, "complete"):
        text = llm.complete(prompt)  # type: ignore[union-attr]
    else:
        raise ConfigurationError(
            "llm must be a callable(prompt)->text or expose .complete(prompt)"
        )
    text = str(text)
    if citations:
        text, removed = sanitize_citations(text, len(result.citations))
        if removed and result.trace is not None:
            result.trace["citations_removed_from_answer"] = removed
    report = None
    if verify is not None:
        report = verify_answer(
            question=question, answer_text=text, context=result.context,
            citations=result.citations, verify=verify, mode=verification_mode,
        )
        text = report.repaired_text
        if result.trace is not None:
            result.trace["verification"] = report.to_dict()
    return Answer(
        text=text,
        context=result.context,
        citations=result.citations,
        result=result,
        verification=report,
    )
