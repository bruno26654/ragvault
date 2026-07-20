"""Preset comparison and retrieval auto-tuning.

Both operate on **retrieval-time** parameters (mode, candidate pool,
ef_search, fusion weights, MMR, chunks per document). Ingestion-time
settings (chunking, embeddings) are not varied because that would require
re-ingesting the corpus — this is stated in the report rather than hidden.

Nothing is ever applied automatically: ``kb.tune()`` returns a
:class:`TuningRecommendation` with evidence, and only an explicit
``kb.apply(recommendation)`` changes the configuration.
"""

from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Iterable, Optional, Union

from .config import PRESETS
from .errors import ConfigurationError, EvaluationError

if TYPE_CHECKING:  # pragma: no cover
    from .evaluate import EvaluationReport
    from .kb import KnowledgeBase

# Retrieval-time knobs a preset is allowed to move during compare/tune.
RETRIEVAL_KNOBS = (
    "retrieval_mode",
    "candidates",
    "ef_search",
    "dense_weight",
    "bm25_weight",
    "sparse_weight",
    "max_chunks_per_document",
    "mmr_lambda",
    "context_window",
)


@dataclass
class ComparisonReport:
    k: int
    reports: dict[str, "EvaluationReport"]
    note: str = (
        "compare() varies retrieval-time parameters only; ingestion-time "
        "settings (chunking, embedding) are those of the existing knowledge base"
    )

    def to_markdown(self) -> str:
        lines = [
            "| preset | recall@k | MRR | nDCG@k | dup rate | ctx tokens | p50 ms | p95 ms |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for name, report in self.reports.items():
            lines.append(
                f"| {name} | {report.recall_at_k:.4f} | {report.mrr:.4f} | "
                f"{report.ndcg_at_k:.4f} | {report.duplicate_rate:.4f} | "
                f"{report.avg_context_tokens:.0f} | {report.latency_p50_ms:.2f} | "
                f"{report.latency_p95_ms:.2f} |"
            )
        lines.append("")
        lines.append(f"> {self.note}")
        return "\n".join(lines)

    def best(self, metric: str = "ndcg_at_k") -> str:
        return max(self.reports, key=lambda n: getattr(self.reports[n], metric))

    def __repr__(self) -> str:
        return f"ComparisonReport(presets={list(self.reports)}, k={self.k})"


@dataclass
class TuningRecommendation:
    objective: str
    k: int
    best_params: dict
    best_score: float
    best_p95_ms: float
    baseline_score: float
    baseline_p95_ms: float
    max_p95_ms: Optional[float]
    trials: list[dict] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            f"## Tuning recommendation (objective: {self.objective}, k={self.k})",
            "",
            f"- baseline: {self.objective}={self.baseline_score:.4f}, "
            f"p95={self.baseline_p95_ms:.2f} ms",
            f"- recommended: {self.objective}={self.best_score:.4f}, "
            f"p95={self.best_p95_ms:.2f} ms",
            f"- params: `{json.dumps(self.best_params, sort_keys=True)}`",
        ]
        if self.max_p95_ms is not None:
            lines.append(f"- latency constraint: p95 <= {self.max_p95_ms} ms")
        lines += ["", "| params | score | p95 ms |", "|---|---|---|"]
        for trial in sorted(self.trials, key=lambda t: -t["score"])[:10]:
            lines.append(
                f"| `{json.dumps(trial['params'], sort_keys=True)}` | "
                f"{trial['score']:.4f} | {trial['p95_ms']:.2f} |"
            )
        lines.append("")
        lines.append(
            "> Evidence: every row above was actually evaluated against the "
            "dataset. Apply with `kb.apply(recommendation)` — never automatic."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"TuningRecommendation({self.objective}={self.best_score:.4f}, "
            f"p95={self.best_p95_ms:.2f}ms, params={self.best_params})"
        )


def _metric_value(report: "EvaluationReport", objective: str) -> float:
    aliases = {
        "ndcg": "ndcg_at_k",
        f"ndcg@{report.k}": "ndcg_at_k",
        "recall": "recall_at_k",
        f"recall@{report.k}": "recall_at_k",
        "mrr": "mrr",
        "hit_rate": "hit_rate",
        "precision": "precision_at_k",
        f"precision@{report.k}": "precision_at_k",
    }
    attr = aliases.get(objective, objective)
    if not hasattr(report, attr):
        raise ConfigurationError(
            f"unknown tuning objective {objective!r}; try 'ndcg@{report.k}', "
            f"'recall@{report.k}', 'mrr', 'hit_rate'"
        )
    return float(getattr(report, attr))


class _ConfigOverride:
    """Temporarily override retrieval knobs on a KB config."""

    def __init__(self, kb: "KnowledgeBase", params: dict) -> None:
        self._kb = kb
        self._params = {k: v for k, v in params.items() if k in RETRIEVAL_KNOBS}
        self._saved: dict = {}

    def __enter__(self) -> None:
        for key, value in self._params.items():
            self._saved[key] = getattr(self._kb.config, key)
            setattr(self._kb.config, key, value)

    def __exit__(self, *exc: object) -> None:
        for key, value in self._saved.items():
            setattr(self._kb.config, key, value)


def compare_presets(
    kb: "KnowledgeBase",
    dataset: Union[str, Path, Iterable[dict]],
    presets: Optional[list[str]] = None,
    *,
    k: int = 10,
) -> ComparisonReport:
    from .evaluate import _load_dataset  # reuse the loader/validation

    rows = _load_dataset(dataset)
    if not rows:
        raise EvaluationError("evaluation dataset is empty")
    presets = presets or ["fast", "balanced", "quality"]
    reports: dict[str, "EvaluationReport"] = {}
    for preset in presets:
        if preset not in PRESETS:
            raise ConfigurationError(
                f"unknown preset {preset!r}; available: {sorted(PRESETS)}"
            )
        overrides = {kk: v for kk, v in PRESETS[preset].items() if kk in RETRIEVAL_KNOBS}
        with _ConfigOverride(kb, overrides):
            reports[preset] = kb.evaluate(rows, k=k)
    return ComparisonReport(k=k, reports=reports)


def tune(
    kb: "KnowledgeBase",
    dataset: Union[str, Path, Iterable[dict]],
    *,
    objective: str = "ndcg@10",
    k: int = 10,
    max_p95_ms: Optional[float] = None,
    grid: Optional[dict[str, list]] = None,
) -> TuningRecommendation:
    """Grid-search retrieval parameters against a dataset.

    The default grid is small on purpose (27 combinations) so tuning stays
    interactive; pass ``grid`` to widen it.
    """
    from .evaluate import _load_dataset

    rows = _load_dataset(dataset)
    if not rows:
        raise EvaluationError("evaluation dataset is empty")

    grid = grid or {
        "ef_search": [48, 96, 192],
        "candidates": [40, 80, 160],
        "bm25_weight": [0.6, 1.0, 1.4],
    }
    for key in grid:
        if key not in RETRIEVAL_KNOBS:
            raise ConfigurationError(
                f"cannot tune {key!r}; tunable knobs: {RETRIEVAL_KNOBS}"
            )

    baseline_report = kb.evaluate(rows, k=k)
    baseline_score = _metric_value(baseline_report, objective)

    trials: list[dict] = []
    keys = sorted(grid)
    best: Optional[dict] = None
    for combo in itertools.product(*(grid[key] for key in keys)):
        params = dict(zip(keys, combo))
        with _ConfigOverride(kb, params):
            report = kb.evaluate(rows, k=k)
        score = _metric_value(report, objective)
        trial = {
            "params": params,
            "score": score,
            "p95_ms": report.latency_p95_ms,
            "recall": report.recall_at_k,
            "mrr": report.mrr,
        }
        trials.append(trial)
        meets_latency = max_p95_ms is None or report.latency_p95_ms <= max_p95_ms
        if meets_latency and (
            best is None
            or score > best["score"]
            or (score == best["score"] and report.latency_p95_ms < best["p95_ms"])
        ):
            best = trial

    if best is None:
        raise EvaluationError(
            f"no configuration met the latency constraint p95 <= {max_p95_ms} ms; "
            f"fastest trial was {min(t['p95_ms'] for t in trials):.2f} ms"
        )
    return TuningRecommendation(
        objective=objective,
        k=k,
        best_params=best["params"],
        best_score=best["score"],
        best_p95_ms=best["p95_ms"],
        baseline_score=baseline_score,
        baseline_p95_ms=baseline_report.latency_p95_ms,
        max_p95_ms=max_p95_ms,
        trials=trials,
    )


def apply_recommendation(kb: "KnowledgeBase", recommendation: TuningRecommendation) -> None:
    """Explicitly apply a recommendation and persist it to ragvault.json."""
    for key, value in recommendation.best_params.items():
        if key not in RETRIEVAL_KNOBS:
            raise ConfigurationError(f"recommendation contains non-tunable key {key!r}")
        setattr(kb.config, key, value)
    config_path = kb.path / "ragvault.json"
    config_path.write_text(
        json.dumps(kb.config.to_dict(), indent=2, sort_keys=True) + "\n"
    )
