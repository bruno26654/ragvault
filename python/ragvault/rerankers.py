"""Reranking utilities.

MaxSim (ColBERT-style late interaction) is offered as a *reranking* stage —
never a naive global multivector search: candidates come from the fused
dense/BM25 pipeline, and only the top-N are rescored with token-level
embeddings.

The token encoder is supplied by you (any callable
``texts -> list of [num_tokens, dim] arrays``); RagVault performs no network
calls and downloads no models.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from .context import RetrievedChunk
from .errors import EmbeddingError

TokenEncoder = Callable[[list[str]], Sequence["np.ndarray"]]


def maxsim_score(query_tokens: "np.ndarray", doc_tokens: "np.ndarray") -> float:
    """MaxSim: for each query token take its best-matching document token,
    sum the maxima. Inputs are [nq, d] and [nd, d] float32 arrays (L2
    normalization is the caller's choice and is respected, not imposed)."""
    q = np.asarray(query_tokens, dtype=np.float32)
    d = np.asarray(doc_tokens, dtype=np.float32)
    if q.ndim != 2 or d.ndim != 2 or q.shape[1] != d.shape[1]:
        raise EmbeddingError(
            f"maxsim expects [n, dim] token matrices with equal dim, "
            f"got {q.shape} and {d.shape}"
        )
    if q.size == 0 or d.size == 0:
        return 0.0
    sims = q @ d.T  # [nq, nd]
    return float(sims.max(axis=1).sum())


def maxsim_reranker(encoder: TokenEncoder, *, batch_size: int = 16):
    """Build a ``rerank`` callable for :meth:`KnowledgeBase.retrieve`.

    ``encoder(texts)`` must return one token-embedding matrix per text.
    Failures propagate to retrieve()'s tolerant rerank handling (previous
    ranking kept, reason recorded in the trace).
    """

    def rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
        if not chunks:
            return chunks
        query_tokens = np.asarray(encoder([query])[0], dtype=np.float32)
        scored: list[tuple[float, RetrievedChunk]] = []
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start:start + batch_size]
            token_mats = encoder([c.text for c in batch])
            if len(token_mats) != len(batch):
                raise EmbeddingError(
                    f"encoder returned {len(token_mats)} matrices for "
                    f"{len(batch)} texts"
                )
            for chunk, mat in zip(batch, token_mats):
                scored.append((maxsim_score(query_tokens, np.asarray(mat)), chunk))
        scored.sort(key=lambda pair: -pair[0])
        out = []
        for score, chunk in scored:
            chunk.score = score
            chunk.selection_reason = "maxsim reranked"
            out.append(chunk)
        return out

    return rerank
