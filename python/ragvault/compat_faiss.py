"""Faiss interoperability (``ragvault.compat.faiss`` namespace).

Compatibility level: **convertible** — vectors and ids are exported to /
imported from Faiss indexes by value. No API, parameter or binary-format
compatibility with Faiss is claimed (see ANALISE-FAISS.md for the
definitions).

Requires the optional ``faiss-cpu`` package (``pip install "ragvault[faiss]"``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional, Sequence

import numpy as np

from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from .kb import KnowledgeBase


def _require_faiss():
    try:
        import faiss
    except ImportError as exc:
        raise ConfigurationError(
            'faiss interop requires faiss-cpu: pip install "ragvault[faiss]"'
        ) from exc
    return faiss


def export_to_faiss(kb: "KnowledgeBase", *, index_factory: Optional[str] = None):
    """Export the KB's live dense vectors into a Faiss index.

    Returns ``(faiss_index, chunk_ids)`` where row *i* of the index is
    ``chunk_ids[i]``. Cosine KBs export normalized vectors into an
    inner-product index (equivalent ranking).
    """
    faiss = _require_faiss()
    chunk_ids, vectors = kb.export_dense()
    if not chunk_ids:
        raise ConfigurationError("knowledge base has no live vectors to export")
    dim = vectors.shape[1]
    metric = kb.config.metric
    if index_factory:
        faiss_metric = (
            faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT
        )
        index = faiss.index_factory(dim, index_factory, faiss_metric)
        if not index.is_trained:
            index.train(vectors)
    elif metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    return index, chunk_ids


def import_vectors(
    kb: "KnowledgeBase",
    ids: Sequence[str],
    vectors: "np.ndarray",
    texts: Optional[Sequence[str]] = None,
    metadata: Optional[Sequence[dict]] = None,
) -> list[str]:
    """Import externally built vectors (e.g. reconstructed from a Faiss
    index) as one-chunk documents with precomputed embeddings.

    The vectors must match the KB's dimension. Texts default to empty
    strings (dense-only rows still participate in dense search; BM25 simply
    has nothing to index for them).
    """
    import json as _json

    vectors = np.ascontiguousarray(np.asarray(vectors, dtype=np.float32))
    if vectors.ndim != 2 or vectors.shape[0] != len(ids):
        raise ConfigurationError(
            f"vectors must be 2D with {len(ids)} rows, got shape {vectors.shape}"
        )
    if vectors.shape[1] != kb.config.dim:
        raise ConfigurationError(
            f"vectors have dimension {vectors.shape[1]}, KB expects {kb.config.dim}"
        )
    out = []
    for i, doc_id in enumerate(ids):
        text = str(texts[i]) if texts is not None else ""
        meta = dict(metadata[i]) if metadata is not None else {}
        document = {
            "document_id": str(doc_id),
            "source_id": "faiss-import",
            "current_version": 1,
            "title": None,
            "metadata": {**meta, "imported": True},
        }
        chunk = {
            "chunk_id": f"{doc_id}#0",
            "document_id": str(doc_id),
            "document_version": 1,
            "chunk_index": 0,
            "text": text,
            "metadata": {},
            "section_path": [],
        }
        kb._vault.upsert_document(
            _json.dumps(document), _json.dumps([chunk]), vectors[i:i + 1]
        )
        out.append(str(doc_id))
    return out


def reconstruct_from_faiss(index, ids: Sequence[str]) -> "np.ndarray":
    """Reconstruct all vectors from a Faiss index that supports it
    (Flat family). Returns float32 [len(ids), dim]."""
    faiss = _require_faiss()
    del faiss  # imported for the dependency check only
    n = index.ntotal
    if n != len(ids):
        raise ConfigurationError(
            f"index has {n} vectors but {len(ids)} ids were provided"
        )
    out = np.zeros((n, index.d), dtype=np.float32)
    for i in range(n):
        out[i] = index.reconstruct(i)
    return out
