"""GPU acceleration via NVIDIA cuVS (optional, experimental).

Status: **implemented, untested on real hardware** — this environment has no
GPU. The wiring (vector export, index build, dense-search override,
post-filter semantics) is covered by unit tests that inject a fake cuvs
module; the cuVS calls themselves follow the documented cuvs Python API and
must be validated on a CUDA machine using the runbook in docs/GPU.md before
being treated as production-ready.

Design:
- The CPU package never imports CUDA. ``pip install cuvs-cu12`` (or the
  RAPIDS channel equivalent) enables this module explicitly.
- :class:`CagraDenseSearcher` is a *sidecar*: it snapshots the KB's live
  dense vectors, builds a CAGRA graph on the GPU and serves dense candidate
  generation. BM25/sparse/fusion/context stay on CPU unchanged.
- Filters are applied as a **post-filter with oversampling** on this path
  (unlike the CPU backends' integrated filtering) — stated in the plan
  RagVault returns, never hidden.
- The searcher is read-only over the snapshot it was built from: documents
  upserted afterwards require ``rebuild()``. ``kb.retrieve`` falls back to
  the CPU path automatically if the GPU search fails (reason recorded in the
  plan).
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from .kb import KnowledgeBase


def _import_cuvs():
    try:
        from cuvs.neighbors import cagra  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ConfigurationError(
            "GPU support requires the optional cuVS package for your CUDA "
            "version, e.g.: pip install cuvs-cu12 (see docs/GPU.md)"
        ) from exc
    return cagra


def is_gpu_available() -> bool:
    """True when cuVS is importable (does not probe device health)."""
    try:
        _import_cuvs()
        return True
    except ConfigurationError:
        return False


class CagraDenseSearcher:
    """CAGRA-backed dense candidate generator (sidecar to a KnowledgeBase).

    Usage::

        searcher = ragvault.gpu.CagraDenseSearcher(kb)   # builds on GPU
        result = kb.retrieve("query", dense_searcher=searcher)

    See module docstring for status and limitations.
    """

    def __init__(
        self,
        kb: "KnowledgeBase",
        *,
        graph_degree: int = 32,
        intermediate_graph_degree: int = 64,
        cagra_module: Any = None,  # injection point for tests
    ) -> None:
        self._cagra = cagra_module or _import_cuvs()
        self._kb = kb
        self._graph_degree = graph_degree
        self._intermediate_graph_degree = intermediate_graph_degree
        self.chunk_ids: list[str] = []
        self._index = None
        self.build_seconds: float = 0.0
        self.rebuild()

    def rebuild(self) -> None:
        """(Re)build the CAGRA graph from the KB's current live vectors."""
        chunk_ids, vectors = self._kb.export_dense()
        if not chunk_ids:
            raise ConfigurationError("knowledge base has no live vectors")
        start = time.monotonic()
        params = self._cagra.IndexParams(
            graph_degree=self._graph_degree,
            intermediate_graph_degree=self._intermediate_graph_degree,
        )
        self._index = self._cagra.build(params, vectors)
        self.build_seconds = time.monotonic() - start
        self.chunk_ids = chunk_ids

    def search(self, query: "np.ndarray", k: int) -> list[tuple[str, float]]:
        """Dense top-k as (chunk_id, similarity score), best first."""
        query = np.ascontiguousarray(np.asarray(query, dtype=np.float32))
        if query.ndim == 1:
            query = query[None, :]
        params = self._cagra.SearchParams()
        distances, neighbors = self._cagra.search(params, self._index, query, k)
        distances = np.asarray(distances)[0]
        neighbors = np.asarray(neighbors)[0]
        out: list[tuple[str, float]] = []
        for dist, idx in zip(distances, neighbors):
            i = int(idx)
            if 0 <= i < len(self.chunk_ids):
                # cuVS returns distances (smaller = closer); RagVault scores
                # are higher-is-better, so negate.
                out.append((self.chunk_ids[i], -float(dist)))
        return out


def dense_override(searcher: CagraDenseSearcher, *, oversample: int = 4):
    """Adapt a searcher into the ``kb.retrieve(dense_searcher=...)`` hook.

    Returns a callable ``(query_vec, pool, accept) -> [(chunk_id, score)]``
    where ``accept`` is the metadata post-filter (applied here with
    ``oversample``x extra candidates — post-filtering, stated in the plan).
    """

    def run(query_vec: "np.ndarray", pool: int, accept) -> list[tuple[str, float]]:
        raw = searcher.search(query_vec, pool * oversample)
        if accept is None:
            return raw[:pool]
        kept = [(cid, score) for cid, score in raw if accept(cid)]
        return kept[:pool]

    return run
