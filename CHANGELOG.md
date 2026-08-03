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

### Fixed
- **A revoked document reached the context looking current.** Revocation was
  judged only *relatively* — a document lost to a better sibling in its
  `doc_group`. When the superseding version was not among the retrieved
  candidates, the group had one member, nothing was compared, and the revoked
  rule was presented as the rule in force with nothing reported. A status in
  the revoked class now eliminates the document by itself. Callers whose own
  filter constrains the status field are exempt: asking for revoked documents
  and then having them deleted for being revoked would undo an explicit
  instruction (the documented "historical facet" pattern).
- **An undecidable precedence conflict was settled by alphabetical order.**
  Documents tying on status, effective date and version fell through to a
  `document_id` tie-break, and the loser's evidence was deleted — so two
  equally current rules that disagreed left a context that looked settled.
  Documents tied at the top all survive now, the conflict is reported with
  `resolved: False` and `tied: [...]`, and `ask_multi` tells the model to
  report the disagreement instead of choosing a side.
- **Eliminating a candidate shrank the context and forfeited a facet's
  coverage slot.** The per-subquery reservation was computed before version
  resolution, so a revoked document that outranked its own replacement took
  the slot, was deleted, and the replacement — just outside the window that
  had been cut for the loser — never arrived. Fusion and resolution now run to
  a fixed point: coverage is re-reserved over what is still eligible and the
  window refills.
- **A citation marker after the sentence terminator was attributed to the
  wrong claim.** Written the ordinary way — `Refunds take 30 days. [1] They
  ship fast. [2]` — the split happened at the period, so `[1]` landed on the
  *next* claim: the claim that actually cited it came back `uncited` (and
  `strict` deleted a sourced fact), the following claim was credited to a
  source it never cited, and the trailing `[2]` became a "claim" that was only
  a marker. Without a space (`days.[1] They`) the answer was not split at all.
  Trailing markers now stay with the claim they source; a marker starting the
  next line still belongs to that line.

### Added
- **A high-demand scenario suite** (`tests/python/test_scenario_versioned_registry.py`):
  ~900 noise documents plus current, revoked, redundant, incomplete and
  partially conflicting rules across four entities. Multi-faceted questions
  whose deadlines, procedures and conditions must not be mixed, with
  deterministic stand-ins that derive everything from the retrieved context.
  Covers retrieval under noise, version resolution, entity separation, direct
  citation support, completeness, claim atomicity, and fail-closed behaviour.
  The three version-resolution bugs above were found by it.
- **`plan["eliminated"]` under `explain=True`** — what was removed and why was
  previously visible only in a full trace.
- **The cited block's text travels with the evidence** (`Citation.text`, and
  `evidence[i]["text"]` in the verification payload). A judge handed only the
  assembled context has to locate `[n]` in it by hand, and can as easily
  justify a claim from a block the claim never cited.
- **`quote`: the one part of a verdict the library can check itself.** A
  verifier may return the span of the cited source that carries the support;
  it is compared against that source (substring, whitespace- and
  case-normalized). A quote in none of the cited sources is fabricated
  attribution — the claim drops to `unsupported` and the discrepancy is
  recorded in `structural_issues`. `require_quotes=True` extends this to
  silence: a `supported` verdict offering no span is not accepted either. Off
  by default because support is not always a contiguous span.
- **`facets=` on `ask()` and `ask_multi()`.** Completeness is judged against
  what the answer *owed*, which is not the same list as the queries retrieval
  ran. A decomposer splitting for search ("policy 2024 revision") was turning
  search terms into answer obligations, each reported as an uncovered facet.
  Declared facets drive both the answer's checklist and the verification, so
  the answer is judged on the obligations it was given; without them
  `ask_multi` still falls back to the subqueries.

### Changed
- **The verifier no longer writes: `replacement` is ignored by default.** A
  verifier that proposes a correction and then re-verifies it is grading its
  own text — self-endorsement, not verification, and the endorsement carries
  the same blind spot that produced the correction. `repair`/`strict` now
  **remove** a claim that does not hold instead of rewriting it. Pass
  `allow_replacements=True` to `ask()`/`ask_multi()` to restore the previous
  behaviour, including the re-verification pass.
- **An unevaluated facet is no longer "unknown".** With facets declared and
  none reported, `complete` was `None`; a caller writing `if complete is not
  False` would ship an answer whose completeness nobody had checked. `None`
  now means only "no facets to cover" — declared-but-unevaluated facets, and a
  verifier that crashed before reporting, both yield `complete=False`. The
  distinction survives in `uncovered_facets[i]["rationale"]`.

### Fixed
- **Verification failed *open*.** `ok` was `not any(problem for claim in
  claims)`, and a crashed verifier leaves `claims` empty — so `not any([])`
  reported the answer as faithful precisely when nothing had been checked.
  Anything gating on `ok` would ship unverified text. Likewise a verifier that
  reported coverage for 1 of 2 facets yielded `complete=True`, silently
  counting the unreported facet as covered. Both now fail closed.

### Added
- **Structural validity as a fourth axis.** `valid` and `structural_issues`
  join fidelity (`ok`), coverage (`complete`) and `segmentation`. A result can
  be structurally unsound regardless of the verdicts, and `ok`/`complete`
  require validity. Issues recorded: facets the verifier never reported on,
  and answer text no claim covered (counted in characters — the library
  reports what went unjudged without guessing whether it was "material").
- **Segmentation structural requirements.** Verifier-supplied spans must be
  verbatim, ordered and **non-overlapping**; overlaps would judge the same
  text twice and make repair produce garbage. Violations are rejected and the
  original answer preserved.
- **Replacements require an explicit `supported`.** Re-verification previously
  accepted anything that was not `unsupported`/`contradicted`, so an `uncited`
  or `inference` verdict let verifier-written text into the answer. Only an
  endorsement counts now.
- **Claim segmentation: scripts without letter case, abbreviations, and
  optional verifier-supplied boundaries.** The split required an uppercase
  letter after the terminator, so answers in Chinese, Japanese, Korean, Arabic
  or Hebrew were **never split at all** — per-claim verification was a no-op
  for every script without case, in a library that ships a multilingual preset.
  CJK/Arabic terminators and case-less scripts are handled now, and an
  abbreviation guard stops `Art. 5º` / `Inc. II` from becoming fragments.
  Heuristics still cannot see two claims inside one sentence, so a verifier may
  now return its own segmentation (claims must be verbatim substrings of the
  answer, keeping repair surgical); `report.segmentation` records which was
  used. This also makes the module docstring true — it already claimed the
  verifier could supply segmentation, but the code rejected any count mismatch.
- **Per-subquery filters** (`subquery_filters=`). A single global filter cannot
  express "the decisional facet needs only current documents, the historical
  facet needs the superseded ones". Each entry replaces the global filter for
  that query (it does not intersect it — otherwise "only REVOGADO" would be
  unreachable under a global "only VIGENTE"); `None` keeps the global filter.
- **Verification: completeness as a separate axis from fidelity.** When the
  question was decomposed, the facets travel in the verifier payload and the
  verifier may report per-facet coverage; the report exposes `facet_coverage`,
  `uncovered_facets` and `complete` — `None` when coverage was not reported,
  because no report is not proof of coverage. An uncovered facet is reported,
  never auto-filled: regenerating would mean an extra LLM call the caller did
  not ask for, with cost and loop risk.
- **Verification: replacements are themselves verified.** A `replacement` was
  generated text entering the answer unchecked. `repair`/`strict` now run
  exactly one extra pass over the replacements (never a loop); a rejected
  replacement is dropped rather than replaced again, `replacement_verdict`
  records the outcome, and a failing recheck keeps the repair while declaring
  it unchecked via `recheck_error`.
- **Verification: original formatting survives a repair.** Rebuilding used
  `" ".join`, which flattened a bulleted or multi-paragraph answer into one
  line whenever a claim was dropped; the original separators are now reused.
  List markers also count as claim boundaries — without that a whole bullet
  list was a single claim, so one bad bullet condemned or spared all of them.
- **Citation metadata.** `Citation.metadata` carries the cited document's
  effective metadata (status, effective date, version), and it travels with
  each `evidence` entry — a verifier could not otherwise tell a current rule
  from a revoked one.
- **Answer facet checklist in `ask_multi`.** The decomposition already
  guaranteed coverage in *retrieval*; it did nothing for coverage in the
  *answer*, so a model handed every required document could still address only
  one facet. The subqueries are now listed in the generation prompt as facets
  the answer must cover. Crucially, the instruction carries an explicit escape
  hatch — a facet with no evidence in the context must be declared unanswered
  rather than guessed — because "do not omit any facet" alone would push the
  model to fabricate exactly where retrieval came up empty. The facets sent are
  recorded in `trace["answer_facets"]`. API unchanged.
- **Post-generation semantic validation** — optional `verify=` callback on
  `ask()` and `ask_multi()`, with `verification_mode` in
  `report`/`annotate`/`repair`/`strict`. Citation-marker sanitizing already
  blocked *invented* `[n]`; this catches the harder failures: a citation that
  exists but does not support its claim, a premise from the user's question
  presented as documented evidence, and a claim contradicted by the source it
  cites. Verdicts are `supported`, `unsupported`, `contradicted`, `uncited`,
  `question_fact` and `inference`. The verifier receives each claim with its
  citations resolved to real chunk ids, and may return a `replacement`.
  A verifier that raises, returns `None` or returns the wrong number of
  verdicts **preserves the original answer** and records the failure; an
  unknown verdict or mode is an actionable `ConfigurationError` rather than a
  silent pass. `answer.verification` and `trace["verification"]` carry claim,
  citations, chunk ids, verdict, rationale, applied action and elapsed time.
  No provider dependency: the verifier is a plain callable.
- **Multi-query pipeline for composed questions** — `kb.retrieve_multi()` and
  `kb.ask_multi()` (plus async variants). Optional question decomposition via
  an external LLM callback or manual `subqueries=`, batched execution through
  the native `search_many`, global Weighted RRF fusion with `chunk_id` dedup,
  optional global reranking, then MMR + context assembly under a single global
  token budget. Neighbor expansion runs only on the final selection, so
  distractors are never expanded.
- **Per-subquery coverage guarantee** (`coverage_per_subquery`, default 1).
  RRF alone loses multi-hop recall: with k0=60 the rank-1 vs rank-10 gap is
  only ~16%, so a document ranking mid-pack for *every* subquery outweighs the
  specialist document that is the top hit for one facet. A benchmark showed the
  fused pool contained every required document for 24/24 composed questions
  while the final context kept them for only 5/24. Reserving each subquery's
  top hits fixes it: **full-recall 0.167 → 0.875**, recall 0.493 → 0.951, at
  1.02x the context tokens (`benchmarks/RESULTS-MULTIQUERY.md`).
- **Document precedence by metadata** (`resolve_versions=True`): status
  (VIGENTE/REVOGADO and English equivalents), effective date, numeric version
  and document type, resolved within a `doc_group`. Superseded versions are
  never dropped silently — they appear in `result.conflicts`, in the trace, and
  are stated in the `ask_multi` prompt. Mandatory `filters=` run as native
  prefilters before search; `boosts=` apply multiplicatively after fusion.
- **Citation integrity**: only retrieved documents can be cited, `[n]` markers
  the context does not contain are stripped from the answer, and the prompt
  states that facts from the question are not documented evidence.
- **Recall-safe reranking**: a reranker may reorder but never drop candidates
  (anything it discards is restored in fused order), and a failing reranker
  falls back to the fused order instead of emptying the result. Likewise a
  failing decomposer falls back to the single original question.
- **Full trace**: subqueries, candidates per subquery, each ranking's
  contribution, fusion results (with `fused_score` and `coverage_reserved`),
  eliminated items with reasons, rerank scores before/after, metadata-filtered
  documents, and per-stage timings.
- Committed multi-hop dataset (24 composed questions: 20 two-hop + 4 three-hop)
  and `benchmarks/bench_multiquery.py`; runnable example
  `examples/multi_query_rag.py` (offline by default, optional `--groq`).

## [1.0.0-rc1] — 2026-07-27

First release candidate for v1.0: CPU single-node, production-hardened. All
release gates in [docs/RELEASE.md](docs/RELEASE.md) are green (full CI matrix
including Linux x86-64/aarch64, macOS Apple Silicon and Windows x86-64 wheels
with clean-install smoke tests, plus real framework integration roundtrips at
pinned versions). Storage v2 (ADR 0016) is complete.

### Added
- **Storage format v2 (binary base + O(delta) delta segments).** The snapshot
  base is written as `gen-N/state.rvseg` — a binary segment container with
  per-record and streaming CRC (`crate::segment`) — and each subsequent flush
  appends an O(delta) binary delta segment (WAL-shaped ops) instead of
  rewriting the base. A bounded delta budget triggers a full base rewrite, and
  `compact()` collapses deltas into a fresh base; open applies base → deltas →
  live WAL. Manifests are `format_version = 2`; v1 vaults migrate transparently.
- **Read-friendly compaction.** The compaction rebuild runs off-lock on a
  cloned snapshot, so concurrent readers keep searching for its whole duration;
  the write lock is held only for the final swap + publish. Concurrent writes
  are detected by seq and never lost (off-lock retry, then an under-lock
  fallback). Completes ADR 0016 — all storage v2 acceptance criteria are met.
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
