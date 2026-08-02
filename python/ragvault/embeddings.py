"""Pluggable embeddings.

The storage core never makes network calls. Embedders are explicit plugins:

- ``"builtin:hashed-ngram"`` (default) — a deterministic, dependency-free
  lexical embedder: hashed character n-grams (3–5) with sublinear tf
  weighting, L2-normalized. It is honest about what it is: a strong lexical
  signal that pairs with BM25 for solid hybrid retrieval fully offline. For
  semantic quality, plug a real model.
- ``"sentence-transformers:<model>"`` — optional extra
  (``pip install "ragvault[local-models]"``); the model is only downloaded
  when you explicitly configure it.
- Any callable ``texts -> ndarray`` or object implementing the
  :class:`Embedder` protocol.
"""

from __future__ import annotations

import hashlib
import json
import math
from typing import Callable, Iterable, Protocol, Sequence, runtime_checkable

import numpy as np

from .errors import ConfigurationError, EmbeddingError


@runtime_checkable
class Embedder(Protocol):
    """Protocol every embedding plugin implements."""

    model_id: str
    dimension: int

    def embed_documents(self, texts: list[str]) -> np.ndarray: ...

    def embed_queries(self, texts: list[str]) -> np.ndarray: ...


class HashedNGramEmbedder:
    """Deterministic offline lexical embedder (hashed char n-grams).

    Not a semantic model — a fast, dependency-free lexical representation
    that makes the default install work end-to-end offline. Documented as
    such everywhere it appears.
    """

    def __init__(self, dimension: int = 512, ngram_range: tuple[int, int] = (3, 5)) -> None:
        if dimension < 8:
            raise ConfigurationError(f"hashed-ngram dimension must be >= 8, got {dimension}")
        self.dimension = dimension
        self.ngram_range = ngram_range
        self.model_id = f"builtin:hashed-ngram:{dimension}"

    def _embed_one(self, text: str) -> np.ndarray:
        vec = np.zeros(self.dimension, dtype=np.float32)
        normalized = " ".join(text.lower().split())
        if not normalized:
            return vec
        lo, hi = self.ngram_range
        counts: dict[int, int] = {}
        data = f"\x02{normalized}\x03"
        for n in range(lo, hi + 1):
            if len(data) < n:
                continue
            for i in range(len(data) - n + 1):
                gram = data[i:i + n]
                h = int.from_bytes(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "little"
                )
                counts[h % self.dimension] = counts.get(h % self.dimension, 0) + 1
        for bucket, count in counts.items():
            vec[bucket] = 1.0 + math.log(count)
        norm = float(np.linalg.norm(vec))
        if norm > 0:
            vec /= norm
        return vec

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return np.stack([self._embed_one(t) for t in texts]) if texts else np.zeros(
            (0, self.dimension), dtype=np.float32
        )

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


class FunctionEmbedder:
    """Wraps a plain callable ``texts -> ndarray`` as an Embedder."""

    def __init__(self, fn: Callable[[list[str]], "np.ndarray"], dimension: int | None = None,
                 model_id: str | None = None) -> None:
        self._fn = fn
        self.model_id = model_id or f"callable:{getattr(fn, '__qualname__', 'embedding')}"
        if dimension is None:
            probe = np.asarray(fn(["dimension probe"]), dtype=np.float32)
            if probe.ndim != 2:
                raise EmbeddingError(
                    f"embedding callable must return a 2D array, got shape {probe.shape}"
                )
            dimension = int(probe.shape[1])
        self.dimension = dimension

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return _validate_output(self._fn(texts), len(texts), self.dimension, self.model_id)

    def embed_queries(self, texts: list[str]) -> np.ndarray:
        return self.embed_documents(texts)


class SentenceTransformersEmbedder:
    """Adapter for sentence-transformers models (optional extra)."""

    def __init__(self, model_name: str) -> None:
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on extras
            raise ConfigurationError(
                f"embedding spec 'sentence-transformers:{model_name}' requires the optional "
                'dependency: pip install "ragvault[local-models]"'
            ) from exc
        self._model = SentenceTransformer(model_name)
        self.model_id = f"sentence-transformers:{model_name}"
        self.dimension = _st_dimension(self._model, self.model_id)

    def embed_documents(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        out = self._model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        return _validate_output(out, len(texts), self.dimension, self.model_id)

    def embed_queries(self, texts: list[str]) -> np.ndarray:  # pragma: no cover
        return self.embed_documents(texts)


def _st_dimension(model: object, model_id: str) -> int:
    """Embedding dimension of a SentenceTransformer, across versions.

    sentence-transformers renamed ``get_sentence_embedding_dimension`` to
    ``get_embedding_dimension``; calling the old name on a new version emits a
    ``FutureWarning``. Prefer the new name, fall back to the old one, and as a
    last resort measure the dimension by encoding a probe string — so a future
    rename cannot break the adapter silently.
    """
    for attr in ("get_embedding_dimension", "get_sentence_embedding_dimension"):
        getter = getattr(model, attr, None)
        if getter is None:
            continue
        dimension = getter()
        if dimension:
            return int(dimension)
    probe = np.asarray(
        model.encode([""], convert_to_numpy=True),  # type: ignore[attr-defined]
        dtype=np.float32,
    )
    if probe.ndim != 2 or probe.shape[1] == 0:
        raise EmbeddingError(
            f"could not determine the embedding dimension of {model_id}"
        )
    return int(probe.shape[1])


def _validate_output(out: object, n: int, dim: int, model_id: str) -> np.ndarray:
    arr = np.asarray(out, dtype=np.float32)
    if arr.shape != (n, dim):
        raise EmbeddingError(
            f"embedder {model_id} returned shape {arr.shape}, expected ({n}, {dim})"
        )
    if not np.all(np.isfinite(arr)):
        raise EmbeddingError(f"embedder {model_id} returned NaN or infinite values")
    return np.ascontiguousarray(arr)


def resolve_embedding(spec: object) -> Embedder:
    """Resolve an embedding spec (string, callable, or Embedder) to an Embedder."""
    if spec is None:
        return HashedNGramEmbedder()
    if isinstance(spec, str):
        if spec.startswith("builtin:hashed-ngram"):
            parts = spec.split(":")
            dim = int(parts[2]) if len(parts) > 2 else 512
            return HashedNGramEmbedder(dimension=dim)
        if spec.startswith("sentence-transformers:"):
            return SentenceTransformersEmbedder(spec.split(":", 1)[1])
        if spec.startswith("local:"):
            # Friendly aliases for common local models via sentence-transformers.
            aliases = {
                "local:multilingual": "paraphrase-multilingual-MiniLM-L12-v2",
                "local:english": "all-MiniLM-L6-v2",
            }
            model = aliases.get(spec)
            if model is None:
                raise ConfigurationError(
                    f"unknown local embedding alias {spec!r}; known: {sorted(aliases)}"
                )
            return SentenceTransformersEmbedder(model)
        raise ConfigurationError(
            f"unknown embedding spec {spec!r}; expected 'builtin:hashed-ngram[:dim]', "
            "'sentence-transformers:<model>', 'local:multilingual', a callable, or an Embedder"
        )
    if callable(spec) and not isinstance(spec, Embedder):
        return FunctionEmbedder(spec)  # type: ignore[arg-type]
    if isinstance(spec, Embedder):
        return spec
    raise ConfigurationError(f"cannot use {type(spec).__name__} as an embedding")


def embedding_fingerprint(embedder: Embedder) -> str:
    """Stable fingerprint of the embedding configuration (cache key part)."""
    payload = json.dumps(
        {"model_id": embedder.model_id, "dimension": embedder.dimension},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def bytes_hash(raw: bytes) -> str:
    """Identity hash of ORIGINAL file bytes. Never derive file identity from
    decoded text (`errors="replace"` collapses distinct binary files)."""
    return hashlib.sha256(raw).hexdigest()


def iter_batches(items: Sequence[str], size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), size):
        yield list(items[i:i + size])
