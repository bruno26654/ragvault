"""The KnowledgeBase — RagVault's main abstraction.

    import ragvault

    kb = ragvault.open("./knowledge", preset="quality")
    kb.sync("./documents")
    result = kb.retrieve("Quais são as regras de cancelamento?", token_budget=6000)
    print(result.context)
    print(result.citations)
"""

from __future__ import annotations

import asyncio
import fnmatch
import json
import sqlite3
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence, Union

import numpy as np

from . import _native
from .chunking import ChunkingConfig, chunk_text, estimate_tokens
from .config import Config
from .context import (
    Citation,
    RetrievalResult,
    RetrievedChunk,
    assemble_context,
)
from .embeddings import (
    Embedder,
    content_hash,
    embedding_fingerprint,
    resolve_embedding,
)
from .errors import ConfigurationError, IngestionError
from .parsers import parse_file, supported_extensions

CONFIG_FILE = "ragvault.json"


@dataclass
class SyncReport:
    discovered: int = 0
    added: int = 0
    updated: int = 0
    unchanged: int = 0
    deleted: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)
    duration_seconds: float = 0.0

    def __repr__(self) -> str:
        return (
            f"SyncReport(discovered={self.discovered}, added={self.added}, "
            f"updated={self.updated}, unchanged={self.unchanged}, "
            f"deleted={self.deleted}, failed={self.failed})"
        )


@dataclass
class Answer:
    text: str
    context: str
    citations: list[Citation]
    result: RetrievalResult

    def __repr__(self) -> str:
        preview = self.text[:80] + ("…" if len(self.text) > 80 else "")
        return f"Answer(text={preview!r}, citations={len(self.citations)})"


class _EmbeddingCache:
    """SQLite-backed embedding cache keyed by (content_hash, model, config).

    Survives reopen, tracks hits/misses, and can be cleared. SQLite is in
    the standard library, so the cache adds no dependencies.
    """

    def __init__(self, path: Path) -> None:
        self._path = path
        self._local = threading.local()
        self.hits = 0
        self.misses = 0
        conn = self._conn()
        conn.execute(
            "CREATE TABLE IF NOT EXISTS embeddings ("
            "key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
        )
        conn.commit()

    def _conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(self._path)
            self._local.conn = conn
        return conn

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        if not keys:
            return {}
        conn = self._conn()
        out: dict[str, np.ndarray] = {}
        for i in range(0, len(keys), 500):
            batch = keys[i:i + 500]
            marks = ",".join("?" for _ in batch)
            for key, dim, blob in conn.execute(
                f"SELECT key, dim, vec FROM embeddings WHERE key IN ({marks})", batch
            ):
                out[key] = np.frombuffer(blob, dtype=np.float32, count=dim)
        self.hits += len(out)
        self.misses += len(keys) - len(out)
        return out

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        if not items:
            return
        conn = self._conn()
        conn.executemany(
            "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
            [(k, len(v), np.asarray(v, dtype=np.float32).tobytes()) for k, v in items.items()],
        )
        conn.commit()

    def clear(self) -> None:
        conn = self._conn()
        conn.execute("DELETE FROM embeddings")
        conn.commit()


def _normalize_documents(
    items: Union[str, Sequence[Union[str, dict]]],
    metadata: Optional[dict],
    ids: Optional[Sequence[str]],
) -> list[dict]:
    if isinstance(items, str):
        items = [items]
    docs: list[dict] = []
    for i, item in enumerate(items):
        if isinstance(item, str):
            doc = {"text": item}
        elif isinstance(item, dict):
            doc = dict(item)
        else:
            raise ConfigurationError(
                f"documents must be strings or dicts, got {type(item).__name__}"
            )
        if "text" not in doc or not str(doc["text"]).strip():
            raise ConfigurationError(f"document at position {i} has no text")
        if ids is not None:
            doc["id"] = ids[i]
        if metadata:
            doc["metadata"] = {**metadata, **doc.get("metadata", {})}
        docs.append(doc)
    return docs


class KnowledgeBase:
    """A local, durable RAG knowledge base."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        preset: str = "balanced",
        embedding: object = None,
        create: bool = True,
        **overrides: object,
    ) -> None:
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        config_path = self.path / CONFIG_FILE

        if config_path.exists():
            stored = json.loads(config_path.read_text())
            self.config = Config.from_dict(stored)
            # Runtime knobs may be overridden per open; identity fields
            # (embedding/dim/metric/chunking) are fixed at creation.
            runtime_fields = {
                "storage", "nprobe", "ef_search", "candidates", "retrieval_mode",
                "flat_threshold", "dense_weight", "bm25_weight", "sparse_weight",
                "default_token_budget", "max_chunks_per_document", "mmr_lambda",
                "context_window", "wal_sync",
            }
            for key, value in overrides.items():
                if value is None:
                    continue
                if key not in runtime_fields:
                    raise ConfigurationError(
                        f"{key!r} is fixed at creation time for an existing "
                        f"knowledge base (runtime-overridable: {sorted(runtime_fields)})"
                    )
                setattr(self.config, key, value)
            if embedding is not None:
                requested = embedding if isinstance(embedding, str) else None
                if requested is not None and requested != self.config.embedding:
                    raise ConfigurationError(
                        f"this knowledge base was created with embedding "
                        f"{self.config.embedding!r} but {requested!r} was requested; "
                        "migrate embeddings instead of silently mixing spaces"
                    )
            spec = embedding if embedding is not None and not isinstance(embedding, str) \
                else self.config.embedding
            self.embedder: Embedder = resolve_embedding(spec)
            if self.embedder.dimension != self.config.dim:
                raise ConfigurationError(
                    f"embedder dimension {self.embedder.dimension} does not match the "
                    f"stored dimension {self.config.dim}"
                )
        else:
            if not create:
                raise ConfigurationError(f"no knowledge base at {self.path}")
            embedding_spec = embedding if embedding is not None else None
            if isinstance(embedding_spec, str):
                overrides = {**overrides, "embedding": embedding_spec}
            self.config = Config.resolve(preset=preset, **overrides)  # type: ignore[arg-type]
            spec = embedding if embedding is not None else self.config.embedding
            self.embedder = resolve_embedding(spec)
            if not isinstance(spec, str):
                self.config.embedding = self.embedder.model_id
            self.config.dim = self.embedder.dimension
            config_path.write_text(
                json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n"
            )

        self._vault = _native.Vault.open(
            str(self.path / "vault"), json.dumps(self.config.engine_config())
        )
        self._cache = _EmbeddingCache(self.path / "embedding-cache.db")
        self._fingerprint = embedding_fingerprint(self.embedder)
        self._tenant_filter: Optional[dict] = None
        self._tenant_metadata: Optional[dict] = None
        self._closed = False

    # -- lifecycle ---------------------------------------------------------

    @classmethod
    def open(cls, path: Union[str, Path], **kwargs: object) -> "KnowledgeBase":
        return cls(path, **kwargs)  # type: ignore[arg-type]

    @classmethod
    def create(cls, path: Union[str, Path], **kwargs: object) -> "KnowledgeBase":
        return cls(path, **kwargs)  # type: ignore[arg-type]

    def close(self) -> None:
        if not self._closed:
            self._vault.close()
            self._closed = True

    def flush(self) -> None:
        self._vault.flush()

    def compact(self) -> None:
        self._vault.compact()

    def __enter__(self) -> "KnowledgeBase":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        state = "closed" if self._closed else "open"
        return f"KnowledgeBase(path={str(self.path)!r}, preset={self.config.preset!r}, {state})"

    # -- ingestion ---------------------------------------------------------

    def _chunking_config(self) -> ChunkingConfig:
        return ChunkingConfig(
            strategy=self.config.chunking_strategy,
            target_tokens=self.config.target_tokens,
            max_tokens=self.config.max_tokens,
            overlap_tokens=self.config.overlap_tokens,
        )

    def _embed_chunks(self, texts: list[str]) -> np.ndarray:
        keys = [f"{content_hash(t)}:{self._fingerprint}" for t in texts]
        cached = self._cache.get_many(keys)
        missing = [i for i, k in enumerate(keys) if k not in cached]
        if missing:
            fresh = self.embedder.embed_documents([texts[i] for i in missing])
            new_items = {keys[i]: fresh[j] for j, i in enumerate(missing)}
            self._cache.put_many(new_items)
            cached.update(new_items)
        return np.stack([cached[k] for k in keys]).astype(np.float32)

    def _upsert_one(self, doc: dict, source_id: Optional[str] = None,
                    uri: Optional[str] = None, fmt: str = "text") -> None:
        doc_id = str(doc.get("id") or content_hash(doc["text"])[:16])
        text = str(doc["text"])
        metadata = dict(doc.get("metadata") or {})
        if self._tenant_metadata:
            metadata.update(self._tenant_metadata)
        metadata["content_hash"] = content_hash(text)
        if uri is not None:
            metadata.setdefault("uri", uri)
        title = doc.get("title")

        raw_chunks = chunk_text(text, self._chunking_config(), fmt=fmt)
        if not raw_chunks:
            raise IngestionError("document produced no chunks", file=doc_id, stage="chunking")
        chunk_dicts = []
        for rc in raw_chunks:
            chunk_dicts.append({
                "chunk_id": f"{doc_id}#{rc.chunk_index}",
                "document_id": doc_id,
                "document_version": 1,  # engine assigns the real version
                "chunk_index": rc.chunk_index,
                "text": rc.text,
                "byte_start": rc.char_start,
                "byte_end": rc.char_end,
                "token_count": rc.token_count,
                "page_number": rc.page_number,
                "section_path": rc.section_path,
                "previous_chunk_id": f"{doc_id}#{rc.chunk_index - 1}" if rc.chunk_index else None,
                "next_chunk_id": (
                    f"{doc_id}#{rc.chunk_index + 1}"
                    if rc.chunk_index + 1 < len(raw_chunks) else None
                ),
                "metadata": {},
            })
        vectors = self._embed_chunks([c["text"] for c in chunk_dicts])
        document = {
            "document_id": doc_id,
            "source_id": source_id,
            "current_version": 1,
            "title": title,
            "metadata": metadata,
        }
        self._vault.upsert_document(
            json.dumps(document), json.dumps(chunk_dicts), vectors
        )

    def add(
        self,
        documents: Union[str, Sequence[Union[str, dict]]],
        *,
        metadata: Optional[dict] = None,
        ids: Optional[Sequence[str]] = None,
    ) -> list[str]:
        """Add raw texts or document dicts. Returns document ids."""
        docs = _normalize_documents(documents, metadata, ids)
        out: list[str] = []
        for doc in docs:
            doc_id = str(doc.get("id") or content_hash(str(doc["text"]))[:16])
            doc["id"] = doc_id
            self._upsert_one(doc)
            out.append(doc_id)
        return out

    def add_documents(self, documents: Sequence[dict]) -> list[str]:
        return self.add(list(documents))

    def remove(self, document_id: str) -> bool:
        return self._vault.delete_document(document_id)

    # -- sync --------------------------------------------------------------

    def sync(
        self,
        directory: Union[str, Path],
        *,
        include: Optional[Sequence[str]] = None,
        exclude: Optional[Sequence[str]] = None,
        delete_missing: bool = True,
        on_error: str = "continue",
    ) -> SyncReport:
        """Idempotently mirror a directory into the knowledge base.

        Unchanged files (by content hash) are skipped; changed files are
        atomically replaced; files that vanished are deleted when
        ``delete_missing`` is set.
        """
        start = time.monotonic()
        directory = Path(directory)
        if not directory.is_dir():
            raise IngestionError(
                f"not a directory: {directory}", stage="discovery",
                suggestion="pass a directory containing your documents",
            )
        report = SyncReport()
        source_id = f"dir:{directory.resolve()}"
        known_exts = supported_extensions()

        files: list[Path] = []
        for path in sorted(directory.rglob("*")):
            if not path.is_file():
                continue
            rel = path.relative_to(directory).as_posix()
            if include and not any(fnmatch.fnmatch(rel, pat) for pat in include):
                continue
            if exclude and any(fnmatch.fnmatch(rel, pat) for pat in exclude):
                continue
            if not include and path.suffix.lower() not in known_exts:
                continue
            files.append(path)
        report.discovered = len(files)

        existing = {
            d["document_id"]: d
            for d in self._vault.list_documents()
            if d.get("source_id") == source_id
        }
        seen_ids: set[str] = set()
        for path in files:
            rel = path.relative_to(directory).as_posix()
            doc_id = rel
            seen_ids.add(doc_id)
            try:
                raw = path.read_bytes()
                file_hash = content_hash(raw.decode("utf-8", errors="replace"))
                prior = existing.get(doc_id)
                if prior and prior.get("metadata", {}).get("file_hash") == file_hash:
                    report.unchanged += 1
                    continue
                parsed = parse_file(path)
                if not parsed.text.strip():
                    report.unchanged += 1
                    continue
                doc = {
                    "id": doc_id,
                    "text": parsed.text,
                    "title": parsed.title,
                    "metadata": {
                        **parsed.metadata,
                        "file_hash": file_hash,
                        "path": rel,
                        "format": parsed.format,
                    },
                }
                self._upsert_one(doc, source_id=source_id, uri=str(path), fmt=parsed.format)
                if prior:
                    report.updated += 1
                else:
                    report.added += 1
            except Exception as exc:
                report.failed += 1
                report.errors.append(f"{rel}: {exc}")
                if on_error == "raise":
                    raise

        if delete_missing:
            for doc_id in set(existing) - seen_ids:
                if self._vault.delete_document(doc_id):
                    report.deleted += 1

        report.duration_seconds = time.monotonic() - start
        return report

    async def async_sync(self, directory: Union[str, Path], **kwargs: Any) -> SyncReport:
        return await asyncio.to_thread(self.sync, directory, **kwargs)

    # -- retrieval ---------------------------------------------------------

    def _merged_filter(self, filters: Optional[dict]) -> Optional[dict]:
        if self._tenant_filter and filters:
            return {"$and": [self._tenant_filter, filters]}
        return self._tenant_filter or filters

    def _hit_to_chunk(self, hit: dict, chunk: dict, doc: Optional[dict]) -> RetrievedChunk:
        metadata = chunk.get("metadata") or {}
        doc_meta = (doc or {}).get("metadata") or {}
        return RetrievedChunk(
            chunk_id=hit["chunk_id"],
            document_id=hit["document_id"],
            document_version=chunk.get("document_version", 1),
            chunk_index=chunk.get("chunk_index", 0),
            text=chunk.get("text", ""),
            score=hit["score"],
            dense_score=hit.get("dense_score"),
            bm25_score=hit.get("bm25_score"),
            sparse_score=hit.get("sparse_score"),
            title=(doc or {}).get("title"),
            uri=doc_meta.get("uri"),
            section_path=chunk.get("section_path") or [],
            page_number=chunk.get("page_number"),
            metadata={**doc_meta, **metadata},
            token_count=chunk.get("token_count") or 0,
        )

    def retrieve(
        self,
        query: str,
        *,
        k: int = 8,
        token_budget: Optional[int] = None,
        filters: Optional[dict] = None,
        mode: Optional[str] = None,
        candidates: Optional[int] = None,
        ef_search: Optional[int] = None,
        nprobe: Optional[int] = None,
        rerank: Optional[Callable[[str, list[RetrievedChunk]], list[RetrievedChunk]]] = None,
        dense_searcher: Optional[object] = None,
        context_window: Optional[dict] = None,
        max_chunks_per_document: Optional[int] = None,
        explain: bool = False,
        trace: bool = False,
    ) -> RetrievalResult:
        """Retrieve model-ready context (not just nearest neighbors)."""
        if not query or not query.strip():
            raise ConfigurationError("query must be a non-empty string")
        mode = mode or self.config.retrieval_mode
        token_budget = token_budget or self.config.default_token_budget
        pool = candidates or self.config.candidates
        merged_filter = self._merged_filter(filters)
        if merged_filter is not None:
            _native.validate_filter(json.dumps(merged_filter))

        trace_data: Optional[dict] = {} if trace else None
        t0 = time.monotonic()
        query_vec = None
        if mode in ("dense", "hybrid", "auto"):
            query_vec = np.ascontiguousarray(
                self.embedder.embed_queries([query])[0], dtype=np.float32
            )
        request = {
            "text": query if mode in ("keyword", "hybrid", "auto") else None,
            "k": max(pool, k),
            "mode": mode,
            "candidates": pool,
            "filter": merged_filter,
            "ef_search": ef_search or self.config.ef_search,
            "nprobe": nprobe or self.config.nprobe,
            "weights": {
                "dense": self.config.dense_weight,
                "bm25": self.config.bm25_weight,
                "sparse": self.config.sparse_weight,
            },
        }
        gpu_note: Optional[str] = None
        if dense_searcher is not None and query_vec is not None:
            try:
                response = self._search_with_dense_override(
                    dense_searcher, query, query_vec, pool, k, mode, merged_filter,
                )
            except Exception as exc:
                gpu_note = f"dense_searcher failed ({exc}); fell back to CPU path"
                response = self._vault.search(json.dumps(request), query_vec)
        else:
            response = self._vault.search(json.dumps(request), query_vec)
        search_ms = (time.monotonic() - t0) * 1000

        doc_cache: dict[str, Optional[dict]] = {}
        chunks: list[RetrievedChunk] = []
        hit_chunks = self._vault.get_chunks([h["chunk_id"] for h in response["hits"]])
        for hit, chunk in zip(response["hits"], hit_chunks):
            if chunk is None:
                continue
            doc_id = hit["document_id"]
            if doc_id not in doc_cache:
                doc_cache[doc_id] = self._vault.get_document(doc_id)
            chunks.append(self._hit_to_chunk(hit, chunk, doc_cache[doc_id]))

        if rerank is not None:
            try:
                chunks = list(rerank(query, chunks))
            except Exception as exc:
                # tolerant mode: keep the pre-rerank order, record why
                if trace_data is not None:
                    trace_data["rerank_error"] = str(exc)
        chunks = chunks[: max(k * 3, k)]

        def fetch_neighbors(chunk: RetrievedChunk, window: dict) -> list[RetrievedChunk]:
            out: list[RetrievedChunk] = []
            doc_chunks = self._vault.get_document_chunks(chunk.document_id)
            by_index = {c["chunk_index"]: c for c in doc_chunks}
            for delta in range(-window.get("before", 0), window.get("after", 0) + 1):
                if delta == 0:
                    continue
                neighbor = by_index.get(chunk.chunk_index + delta)
                if neighbor and neighbor.get("document_version") == chunk.document_version:
                    out.append(
                        self._hit_to_chunk(
                            {"chunk_id": neighbor["chunk_id"],
                             "document_id": chunk.document_id,
                             "score": 0.0},
                            neighbor,
                            self._vault.get_document(chunk.document_id),
                        )
                    )
            return out

        result = assemble_context(
            chunks[: k * 3],
            token_budget=token_budget,
            max_chunks=k,
            max_chunks_per_document=(
                max_chunks_per_document or self.config.max_chunks_per_document
            ),
            mmr_lambda=self.config.mmr_lambda,
            fetch_neighbors=fetch_neighbors,
            context_window=context_window or self.config.context_window,
            trace=trace_data,
        )
        result.plan = response.get("plan", {})
        if gpu_note is not None:
            result.plan = dict(result.plan)
            result.plan.setdefault("reason", [])
            result.plan["reason"] = list(result.plan["reason"]) + [gpu_note]
        if explain or trace:
            result.plan = dict(result.plan)
            result.plan["search_ms"] = round(search_ms, 3)
            result.plan["mode"] = mode
            result.plan["token_budget"] = token_budget
        if trace_data is not None:
            trace_data["search_ms"] = round(search_ms, 3)
            result.trace = trace_data
        return result

    def _search_with_dense_override(
        self,
        searcher: object,
        query: str,
        query_vec: np.ndarray,
        pool: int,
        k: int,
        mode: str,
        merged_filter: Optional[dict],
    ) -> dict:
        """Dense candidates from a sidecar searcher (e.g. GPU CAGRA), BM25
        from the engine, weighted-RRF fusion in Python. Filters on the dense
        side are a post-filter with oversampling — recorded in the plan."""
        raw = searcher.search(query_vec, pool * 4)
        if merged_filter is not None and raw:
            mask = self._vault.filter_chunks(
                [cid for cid, _ in raw], json.dumps(merged_filter)
            )
            raw = [pair for pair, ok in zip(raw, mask) if ok]
        dense = raw[:pool]

        bm25: list[tuple[str, float]] = []
        if mode in ("hybrid", "auto", "keyword"):
            response = self._vault.search(json.dumps({
                "text": query, "k": pool, "mode": "keyword",
                "candidates": pool, "filter": merged_filter,
            }))
            bm25 = [(h["chunk_id"], h["bm25_score"]) for h in response["hits"]]

        fused: dict[str, float] = {}
        for weight, ranked in (
            (self.config.dense_weight, dense),
            (self.config.bm25_weight, bm25),
        ):
            for rank, (cid, _) in enumerate(ranked):
                fused[cid] = fused.get(cid, 0.0) + weight / (60.0 + rank + 1.0)
        ordered = sorted(fused.items(), key=lambda kv: (-kv[1], kv[0]))[: max(pool, k)]
        dense_map = dict(dense)
        bm25_map = dict(bm25)
        hits = []
        for cid, score in ordered:
            doc_id = cid.rsplit("#", 1)[0]
            hits.append({
                "chunk_id": cid, "document_id": doc_id, "score": score,
                "dense_score": dense_map.get(cid), "bm25_score": bm25_map.get(cid),
                "sparse_score": None, "internal_id": 0,
            })
        return {
            "hits": hits,
            "plan": {
                "mode": mode,
                "dense_backend": "gpu_sidecar",
                "reason": [
                    "dense candidates from external searcher "
                    f"({type(searcher).__name__})",
                    "filters applied as post-filter with 4x oversampling on "
                    "the dense side (integrated filtering stays on the CPU "
                    "backends)",
                ],
                "candidate_pool": pool,
                "filtered": merged_filter is not None,
            },
        }

    def retrieve_many(self, queries: Sequence[str], **kwargs: Any) -> list[RetrievalResult]:
        return [self.retrieve(q, **kwargs) for q in queries]

    async def aretrieve(self, query: str, **kwargs: Any) -> RetrievalResult:
        return await asyncio.to_thread(self.retrieve, query, **kwargs)

    async def aretrieve_many(self, queries: Sequence[str], **kwargs: Any) -> list[RetrievalResult]:
        return await asyncio.gather(*(self.aretrieve(q, **kwargs) for q in queries))

    # -- ask ---------------------------------------------------------------

    def ask(
        self,
        question: str,
        *,
        llm: Callable[[str], str],
        citations: bool = True,
        system_prompt: Optional[str] = None,
        **retrieve_kwargs: Any,
    ) -> Answer:
        """Retrieve context and call a user-provided LLM. The LLM is a plain
        callable ``prompt -> answer text``; RagVault never calls external
        services on its own."""
        result = self.retrieve(question, **retrieve_kwargs)
        instructions = system_prompt or (
            "Answer the question using only the context below. "
            + ("Cite sources with [n] markers matching the context blocks. "
               if citations else "")
            + "If the context does not contain the answer, say so."
        )
        prompt = (
            f"{instructions}\n\n# Context\n{result.context}\n\n"
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
        return Answer(
            text=str(text),
            context=result.context,
            citations=result.citations,
            result=result,
        )

    # -- evaluation --------------------------------------------------------

    def evaluate(self, dataset: Union[str, Path, Iterable[dict]], *, k: int = 10,
                 **retrieve_kwargs: Any):
        from .evaluate import evaluate_kb

        return evaluate_kb(self, dataset, k=k, **retrieve_kwargs)

    def compare(self, dataset: Union[str, Path, Iterable[dict]],
                presets: Optional[list[str]] = None, *, k: int = 10):
        """Evaluate several presets' retrieval-time settings on this KB."""
        from .tuning import compare_presets

        return compare_presets(self, dataset, presets, k=k)

    def tune(self, dataset: Union[str, Path, Iterable[dict]], *,
             objective: str = "ndcg@10", k: int = 10,
             max_p95_ms: Optional[float] = None,
             grid: Optional[dict] = None):
        """Grid-search retrieval parameters; returns a recommendation with
        evidence. Never applies anything automatically."""
        from .tuning import tune as _tune

        return _tune(self, dataset, objective=objective, k=k,
                     max_p95_ms=max_p95_ms, grid=grid)

    def apply(self, recommendation) -> None:
        """Explicitly apply a TuningRecommendation and persist the config."""
        from .tuning import apply_recommendation

        apply_recommendation(self, recommendation)

    def export_dense(self) -> tuple[list[str], np.ndarray]:
        """Export live dense vectors as (chunk_ids, float32 [n, dim])."""
        ids, flat, dim = self._vault.export_dense()
        vectors = np.asarray(flat, dtype=np.float32).reshape(-1, dim) if ids else \
            np.zeros((0, self.config.dim or 0), dtype=np.float32)
        return ids, np.ascontiguousarray(vectors)

    # -- embedding migration -------------------------------------------------

    def migrate_embeddings(self, new_embedding: object, *,
                           strategy: str = "blocking") -> None:
        """Re-embed every current document version with a new embedder and
        atomically swap the vault.

        Only ``strategy="blocking"`` is implemented (background/copy-on-write
        are planned). The old vault stays intact until the new one is fully
        built, flushed and validated; the swap is two directory renames.
        Caveats (documented): version history and user-supplied sparse
        vectors are not carried over — only current document versions.
        """
        if self._tenant_filter is not None:
            raise ConfigurationError(
                "migrate_embeddings must run on the base KnowledgeBase, "
                "not a tenant view"
            )
        if strategy != "blocking":
            raise ConfigurationError(
                f"strategy {strategy!r} not implemented; available: 'blocking'"
            )
        new_embedder = resolve_embedding(new_embedding)
        if (new_embedder.model_id == self.embedder.model_id
                and new_embedder.dimension == self.embedder.dimension):
            return  # nothing to do

        import shutil

        new_config = Config.from_dict(self.config.to_dict())
        new_config.embedding = (
            new_embedding if isinstance(new_embedding, str) else new_embedder.model_id
        )
        new_config.dim = new_embedder.dimension
        new_fingerprint = embedding_fingerprint(new_embedder)

        tmp_vault = self.path / "vault-migrate.tmp"
        if tmp_vault.exists():
            shutil.rmtree(tmp_vault)
        new_vault = _native.Vault.open(
            str(tmp_vault), json.dumps(new_config.engine_config())
        )
        try:
            for doc in self._vault.list_documents():
                chunks = self._vault.get_document_chunks(doc["document_id"])
                if not chunks:
                    continue
                texts = [c["text"] for c in chunks]
                keys = [f"{content_hash(t)}:{new_fingerprint}" for t in texts]
                cached = self._cache.get_many(keys)
                missing = [i for i, key in enumerate(keys) if key not in cached]
                if missing:
                    fresh = new_embedder.embed_documents([texts[i] for i in missing])
                    new_items = {keys[i]: fresh[j] for j, i in enumerate(missing)}
                    self._cache.put_many(new_items)
                    cached.update(new_items)
                vectors = np.stack([cached[key] for key in keys]).astype(np.float32)
                new_vault.upsert_document(
                    json.dumps(doc), json.dumps(chunks), np.ascontiguousarray(vectors)
                )
            new_vault.flush()
        except Exception:
            new_vault.close()
            shutil.rmtree(tmp_vault, ignore_errors=True)
            raise  # old vault untouched
        new_vault.close()

        # Swap: old vault preserved until the new one is in place.
        self._vault.close()
        old_vault = self.path / "vault-old.tmp"
        if old_vault.exists():
            shutil.rmtree(old_vault)
        (self.path / "vault").rename(old_vault)
        tmp_vault.rename(self.path / "vault")
        shutil.rmtree(old_vault, ignore_errors=True)

        self.config = new_config
        (self.path / CONFIG_FILE).write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n"
        )
        self.embedder = new_embedder
        self._fingerprint = new_fingerprint
        self._vault = _native.Vault.open(
            str(self.path / "vault"), json.dumps(self.config.engine_config())
        )

    # -- integrations --------------------------------------------------------

    def as_langchain_retriever(self, *, k: int = 8, **retrieve_kwargs: Any):
        from .integrations import as_langchain_retriever

        return as_langchain_retriever(self, k=k, **retrieve_kwargs)

    def as_llamaindex_retriever(self, *, k: int = 8, **retrieve_kwargs: Any):
        from .integrations import as_llamaindex_retriever

        return as_llamaindex_retriever(self, k=k, **retrieve_kwargs)

    def as_haystack_retriever(self, *, k: int = 8, **retrieve_kwargs: Any):
        from .integrations import as_haystack_retriever

        return as_haystack_retriever(self, k=k, **retrieve_kwargs)

    def as_dspy_retriever(self, *, k: int = 8, **retrieve_kwargs: Any):
        from .integrations import as_dspy_retriever

        return as_dspy_retriever(self, k=k, **retrieve_kwargs)

    # -- multi-tenancy -----------------------------------------------------

    def for_tenant(self, tenant_id: str) -> "KnowledgeBase":
        """A view that automatically tags writes and filters every query with
        ``tenant_id``. No query through this view can see other tenants."""
        view = object.__new__(KnowledgeBase)
        view.__dict__ = {**self.__dict__}
        view._tenant_filter = {"tenant_id": tenant_id}
        view._tenant_metadata = {"tenant_id": tenant_id}
        return view

    # -- introspection -----------------------------------------------------

    def stats(self) -> dict:
        stats = self._vault.stats()
        stats["embedding_cache"] = {"hits": self._cache.hits, "misses": self._cache.misses}
        stats["embedding"] = self.embedder.model_id
        return stats

    def inspect(self, document_id: str) -> dict:
        doc = self._vault.get_document(document_id)
        if doc is None:
            raise ConfigurationError(f"document not found: {document_id}")
        return {
            "document": doc,
            "chunks": self._vault.get_document_chunks(document_id),
            "versions": self._vault.list_document_versions(document_id),
        }

    def documents(self) -> list[dict]:
        return self._vault.list_documents()


def open_kb(path: Union[str, Path], **kwargs: object) -> KnowledgeBase:
    """Module-level ``ragvault.open()``."""
    return KnowledgeBase(path, **kwargs)  # type: ignore[arg-type]
