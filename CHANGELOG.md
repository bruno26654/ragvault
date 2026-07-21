# Changelog

All notable changes to RagVault are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project
adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

While the version is `0.x`, the public Python API surface described in
[docs/PYTHON-API.md](docs/PYTHON-API.md) is treated as stable and changes to
it are called out under **Changed** with a migration note; internal Rust
crate APIs and the on-disk format may still evolve behind the documented
compatibility guarantees (see [docs/STORAGE.md](docs/STORAGE.md)).

## [Unreleased]

### Added
- **Storage format v2 (binary base + O(delta) delta segments).** The snapshot
  base is written as `gen-N/state.rvseg` — a binary segment container with
  per-record and streaming CRC (`crate::segment`) — and each subsequent flush
  appends an O(delta) binary delta segment (WAL-shaped ops) instead of
  rewriting the base. A bounded delta budget triggers a full base rewrite, and
  `compact()` collapses deltas into a fresh base; open applies base → deltas →
  live WAL. Manifests are `format_version = 2`; v1 vaults migrate transparently.
  The one remaining v2 follow-up is non-blocking concurrent compaction (ADR
  0016) — current compaction is synchronous but always crash- and read-safe.
- **Native batch retrieval.** `kb.retrieve_many()` / `kb.search_many()` run a
  list of queries through one GIL-released Rust call with per-query
  parallelism, returning results identical to sequential `retrieve()`
  (`TestNativeBatch`).
- **Typed metadata indexes.** Keyword/boolean posting lists plus a numeric
  `BTreeMap` for range predicates, intersected for `AND` filters with a
  residual predicate for partial coverage. Visible in the query plan and
  benchmarked at 12× (10% selectivity), 172× (1%) and 484× (0.1%) over the
  scan baseline (`benchmarks/RESULTS-FILTERS.md`).
- **Reproducible real-text RAG evaluation harness.** Committed dataset
  (30 passages / 24 queries, 12 paraphrase + 12 keyword) and
  `benchmarks/bench_rag_quality.py` reporting Recall@k, MRR, nDCG, precision,
  duplication, token counts and p50/p95 for bm25 / lexical / hybrid / +MMR /
  +expansion (`benchmarks/RESULTS-RAG.md`).
- **Context builder v2.** Adjacent retrieved runs are merged, and
  `result.truncated` explicitly signals token-budget truncation.
- **Real framework integration tests in CI.** LangChain, LlamaIndex, Haystack
  and DSPy adapters are exercised against pinned upstream versions in a
  dedicated CI job (`TestIntegrations`).
- **Multiplatform wheel matrix.** CI builds wheels for Linux x86-64, Linux
  aarch64, macOS Apple Silicon (macos-14) and Windows x86-64, each followed by
  a clean-venv install smoke test of the `open → add → retrieve` path.

### Changed
- **Honest presets.** `preset="quality"` now requires an explicit embedding
  and refuses to fall back to a lexical baseline silently; `offline-lite` is
  the named, documented hashed-ngram baseline for no-download environments.
- **Per-query IVF `nprobe`.** `nprobe` is now a per-request parameter
  (`kb.retrieve(..., nprobe=N)`) instead of being pinned by the stored config
  on reopen, restoring the expected recall/`nprobe` curve.

### Fixed
- **Windows write-path portability.** The WAL is opened read+write (seeked to
  end) instead of in append mode, so `set_len` no longer fails with "Access is
  denied" on `close()` under Windows; and `KnowledgeBase.close()` now releases
  the sqlite embedding-cache handle so the knowledge base directory can be
  removed. Both were caught by the multiplatform wheel smoke test.
- **P0 write-path atomicity.** Prepared writes are now validated in full
  *before* the WAL append; apply is infallible and publishes at the end, and a
  corrupt batch fails replay as explicit corruption instead of leaving a
  partial mutation or a poisoned WAL. Reproduced first by
  `rejected_write_leaves_no_trace_even_after_reopen` and
  `rejected_replace_preserves_old_version`, then fixed.
- **Source identity by raw bytes.** Ingestion now keys source identity on
  `sha256(raw_bytes)` with separate parsed-content and processing-pipeline
  fingerprints, so binary-distinct sources are never conflated and a pipeline
  change correctly invalidates cached processing (`TestSourceIdentity`).

## [0.1.0] — Rust core + Python knowledge base

Initial pre-release of the RagVault engine and Python API.

### Added
- Rust workspace: `ragvault-core` (types, error taxonomy, filter DSL),
  `ragvault-vector` (flat, HNSW, SQ8, IVF-Flat, IVF-PQ kernels),
  `ragvault-retrieval` (BM25, sparse, weighted RRF fusion),
  `ragvault-engine` (WAL, snapshot generations, planner, compaction) and
  `ragvault-python` (PyO3 abi3-py39 bindings).
- Durable persistence: CRC32 WAL with torn-tail truncation, snapshot
  generations with atomic publish, single-writer lock, and mmap storage
  (`storage="mmap"`).
- Python `KnowledgeBase` API: `open`/`add`/`sync`/`retrieve`/`ask`/`evaluate`,
  structural chunking, pluggable embeddings with a sqlite cache, MMR
  diversity, neighbor expansion, token budgeting and stable citations.
- Presets, multi-tenancy with adversarial isolation, CLI
  (`init`/`sync`/`query`/`inspect`/`doctor`/`evaluate`/`compact`), Studio UI,
  pluggable reranking (including MaxSim late interaction), `kb.compare` /
  `kb.tune` / `kb.apply`, Faiss interop and an experimental cuVS/CAGRA GPU
  sidecar (unvalidated on real hardware — see [docs/GPU.md](docs/GPU.md)).
