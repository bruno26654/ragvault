"""Configuration and quality presets.

Every automatic choice is explicit, inspectable (``kb.config.explain()``)
and exportable (``kb.config.export(path)``). Presets set defaults; explicit
keyword arguments always win.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

from .errors import ConfigurationError

PRESETS: dict[str, dict] = {
    "balanced": {},
    "quality": {
        "retrieval_mode": "hybrid",
        "candidates": 100,
        "ef_search": 128,
        "max_chunks_per_document": 3,
        "context_window": {"before": 1, "after": 1},
        "mmr_lambda": 0.7,
        "target_tokens": 350,
    },
    "fast": {
        "retrieval_mode": "hybrid",
        "candidates": 40,
        "ef_search": 48,
        "max_chunks_per_document": 2,
        "context_window": {"before": 0, "after": 0},
        "target_tokens": 450,
    },
    "offline": {
        "embedding": "builtin:hashed-ngram",
        "retrieval_mode": "hybrid",
    },
    "multilingual": {
        "embedding": "local:multilingual",
        "retrieval_mode": "hybrid",
    },
    "code": {
        "chunking_strategy": "auto",
        "target_tokens": 300,
        "retrieval_mode": "hybrid",
        "bm25_weight": 1.2,
    },
    "long_documents": {
        "target_tokens": 500,
        "max_tokens": 900,
        "context_window": {"before": 1, "after": 1},
        "max_chunks_per_document": 4,
    },
    "high_recall": {
        "candidates": 200,
        "ef_search": 256,
        "retrieval_mode": "hybrid",
    },
    "low_memory": {
        "candidates": 30,
        "ef_search": 32,
        "hnsw_m": 8,
        "quantization": "sq8",
    },
}


@dataclass
class Config:
    """Full knowledge-base configuration (persisted in ragvault.json)."""

    preset: str = "balanced"
    embedding: str = "builtin:hashed-ngram"
    dim: Optional[int] = None  # resolved from the embedder on create
    metric: str = "cosine"
    # chunking
    chunking_strategy: str = "auto"
    target_tokens: int = 400
    max_tokens: int = 700
    overlap_tokens: int = 40
    # index
    hnsw_m: int = 16
    hnsw_ef_construction: int = 200
    ef_search: int = 64
    flat_threshold: int = 1000
    quantization: str = "none"  # "none" | "sq8" (int8 scan + f32 rescore)
    # retrieval
    retrieval_mode: str = "hybrid"
    candidates: int = 80
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    sparse_weight: float = 1.0
    # context assembly
    default_token_budget: int = 4000
    max_chunks_per_document: int = 3
    mmr_lambda: float = 0.7
    context_window: dict = field(default_factory=lambda: {"before": 0, "after": 0})
    citation_format: str = "inline"
    # durability
    wal_sync: str = "batch"

    @classmethod
    def resolve(cls, preset: str = "balanced", **overrides: object) -> "Config":
        if preset not in PRESETS:
            raise ConfigurationError(
                f"unknown preset {preset!r}; available: {sorted(PRESETS)}"
            )
        merged: dict = {"preset": preset, **PRESETS[preset]}
        for key, value in overrides.items():
            if value is None:
                continue
            if not hasattr(cls, "__dataclass_fields__") or key not in cls.__dataclass_fields__:
                raise ConfigurationError(f"unknown configuration option {key!r}")
            merged[key] = value
        return cls(**merged)  # type: ignore[arg-type]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Config":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(**known)

    def export(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n")

    def explain(self) -> str:
        lines = [
            f"preset: {self.preset}",
            f"embedding: {self.embedding} (dim={self.dim})",
            f"metric: {self.metric}",
            "chunking: strategy={0} target={1} max={2} overlap={3} tokens".format(
                self.chunking_strategy, self.target_tokens, self.max_tokens,
                self.overlap_tokens,
            ),
            "index: hnsw(M={0}, ef_construction={1}, ef_search={2}), "
            "flat below {3} vectors, quantization={4}".format(
                self.hnsw_m, self.hnsw_ef_construction, self.ef_search,
                self.flat_threshold, self.quantization,
            ),
            "retrieval: mode={0} candidate_pool={1} weights(dense={2}, bm25={3}, "
            "sparse={4})".format(
                self.retrieval_mode, self.candidates, self.dense_weight,
                self.bm25_weight, self.sparse_weight,
            ),
            "context: budget={0} tokens, max {1} chunks/document, mmr_lambda={2}, "
            "window={3}".format(
                self.default_token_budget, self.max_chunks_per_document,
                self.mmr_lambda, self.context_window,
            ),
            f"durability: wal_sync={self.wal_sync}",
        ]
        return "\n".join(lines)

    def engine_config(self) -> dict:
        if not self.dim:
            raise ConfigurationError("dimension not resolved yet (open the KB first)")
        return {
            "dim": self.dim,
            "metric": self.metric,
            "hnsw": {
                "m": self.hnsw_m,
                "ef_construction": self.hnsw_ef_construction,
                "ef_search": self.ef_search,
                "seed": 0x52616756,
            },
            "bm25": {"k1": 1.2, "b": 0.75, "lowercase": True},
            "wal_sync": self.wal_sync,
            "flat_threshold": self.flat_threshold,
            "quantization": self.quantization,
        }
