"""Native retrieval evaluation.

Dataset format (JSONL or an iterable of dicts):

    {"query": "how do refunds work?", "relevant_ids": ["policies.md"]}

``relevant_ids`` are document ids (chunk-level ids also work). Metrics are
retrieval metrics — answer-generation metrics belong to optional LLM
integrations, not the core.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Iterable, Union

from .errors import EvaluationError

if TYPE_CHECKING:  # pragma: no cover
    from .kb import KnowledgeBase


@dataclass
class EvaluationReport:
    queries: int
    k: int
    recall_at_k: float
    precision_at_k: float
    hit_rate: float
    mrr: float
    ndcg_at_k: float
    document_recall: float
    duplicate_rate: float
    avg_context_tokens: float
    latency_p50_ms: float
    latency_p95_ms: float
    per_query: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "queries": self.queries,
            "k": self.k,
            "recall_at_k": self.recall_at_k,
            "precision_at_k": self.precision_at_k,
            "hit_rate": self.hit_rate,
            "mrr": self.mrr,
            "ndcg_at_k": self.ndcg_at_k,
            "document_recall": self.document_recall,
            "duplicate_rate": self.duplicate_rate,
            "avg_context_tokens": self.avg_context_tokens,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p95_ms": self.latency_p95_ms,
        }

    def to_json(self, path: Union[str, Path, None] = None) -> str:
        payload = json.dumps({**self.to_dict(), "per_query": self.per_query}, indent=2)
        if path:
            Path(path).write_text(payload + "\n")
        return payload

    def to_markdown(self) -> str:
        rows = [
            ("queries", self.queries),
            (f"recall@{self.k}", f"{self.recall_at_k:.4f}"),
            (f"precision@{self.k}", f"{self.precision_at_k:.4f}"),
            ("hit rate", f"{self.hit_rate:.4f}"),
            ("MRR", f"{self.mrr:.4f}"),
            (f"nDCG@{self.k}", f"{self.ndcg_at_k:.4f}"),
            ("document recall", f"{self.document_recall:.4f}"),
            ("duplicate rate", f"{self.duplicate_rate:.4f}"),
            ("avg context tokens", f"{self.avg_context_tokens:.1f}"),
            ("latency p50 (ms)", f"{self.latency_p50_ms:.2f}"),
            ("latency p95 (ms)", f"{self.latency_p95_ms:.2f}"),
        ]
        lines = ["| metric | value |", "|---|---|"]
        lines += [f"| {name} | {value} |" for name, value in rows]
        return "\n".join(lines)

    def to_csv(self, path: Union[str, Path, None] = None) -> str:
        data = self.to_dict()
        text = ",".join(data.keys()) + "\n" + ",".join(str(v) for v in data.values()) + "\n"
        if path:
            Path(path).write_text(text)
        return text

    def __repr__(self) -> str:
        return (
            f"EvaluationReport(queries={self.queries}, recall@{self.k}="
            f"{self.recall_at_k:.3f}, mrr={self.mrr:.3f}, ndcg@{self.k}="
            f"{self.ndcg_at_k:.3f})"
        )


def _load_dataset(dataset: Union[str, Path, Iterable[dict]]) -> list[dict]:
    if isinstance(dataset, (str, Path)):
        path = Path(dataset)
        if not path.exists():
            raise EvaluationError(f"evaluation dataset not found: {path}")
        rows = []
        for i, line in enumerate(path.read_text().splitlines()):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise EvaluationError(f"invalid JSONL at {path}:{i + 1}: {exc}") from exc
        return rows
    return list(dataset)


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return ordered[idx]


def evaluate_kb(
    kb: "KnowledgeBase",
    dataset: Union[str, Path, Iterable[dict]],
    *,
    k: int = 10,
    **retrieve_kwargs: Any,
) -> EvaluationReport:
    rows = _load_dataset(dataset)
    if not rows:
        raise EvaluationError("evaluation dataset is empty")

    recalls, precisions, hits, mrrs, ndcgs, doc_recalls = [], [], [], [], [], []
    dup_rates, token_counts, latencies = [], [], []
    per_query = []
    for row in rows:
        query = row.get("query")
        relevant = set(map(str, row.get("relevant_ids", [])))
        if not query or not relevant:
            raise EvaluationError(
                f"dataset rows need 'query' and 'relevant_ids': {row!r}"
            )
        t0 = time.monotonic()
        result = kb.retrieve(query, k=k, **retrieve_kwargs)
        latency = (time.monotonic() - t0) * 1000
        latencies.append(latency)

        retrieved_docs = []
        for chunk in result.chunks:
            if chunk.expanded:
                continue
            retrieved_docs.append(chunk.document_id)
        # id match at document or chunk granularity
        def is_relevant(chunk_doc: str, chunk_id: str) -> bool:
            return chunk_doc in relevant or chunk_id in relevant

        flags = [
            is_relevant(c.document_id, c.chunk_id)
            for c in result.chunks if not c.expanded
        ][:k]
        found_docs = {d for d in retrieved_docs if d in relevant}

        recall = len(found_docs) / len(relevant)
        precision = (sum(flags) / len(flags)) if flags else 0.0
        hit = 1.0 if any(flags) else 0.0
        rr = 0.0
        for rank, flag in enumerate(flags, start=1):
            if flag:
                rr = 1.0 / rank
                break
        dcg = sum(f / math.log2(rank + 1) for rank, f in enumerate(flags, start=1))
        ideal = sum(
            1.0 / math.log2(rank + 1)
            for rank in range(1, min(len(relevant), k) + 1)
        )
        ndcg = dcg / ideal if ideal > 0 else 0.0
        unique_docs = len(set(retrieved_docs))
        dup_rate = 1.0 - unique_docs / len(retrieved_docs) if retrieved_docs else 0.0

        recalls.append(recall)
        precisions.append(precision)
        hits.append(hit)
        mrrs.append(rr)
        ndcgs.append(ndcg)
        doc_recalls.append(recall)
        dup_rates.append(dup_rate)
        token_counts.append(result.token_count)
        per_query.append({
            "query": query,
            "recall": recall,
            "mrr": rr,
            "ndcg": ndcg,
            "latency_ms": round(latency, 2),
            "retrieved": retrieved_docs[:k],
        })

    return EvaluationReport(
        queries=len(rows),
        k=k,
        recall_at_k=statistics.mean(recalls),
        precision_at_k=statistics.mean(precisions),
        hit_rate=statistics.mean(hits),
        mrr=statistics.mean(mrrs),
        ndcg_at_k=statistics.mean(ndcgs),
        document_recall=statistics.mean(doc_recalls),
        duplicate_rate=statistics.mean(dup_rates),
        avg_context_tokens=statistics.mean(token_counts),
        latency_p50_ms=_percentile(latencies, 0.50),
        latency_p95_ms=_percentile(latencies, 0.95),
        per_query=per_query,
    )
