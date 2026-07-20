# TASKS

Rastreabilidade de requisitos → implementação → evidência. Estados:
`not-started | in-progress | blocked | implemented | under-review | validated | experimental | deferred`

> **Auditoria (sessão de hardening, commit-base f69fc5e):** baseline real
> re-executado — fmt/clippy limpos; testes Rust (95 unit/lib + 6 differential
> + 5 proptest) e 79 Python verdes; 1 teste `gpu` deselected (sem hardware).
> **P0 de atomicidade foi encontrado, reproduzido e corrigido** (5aec3b9):
> até então o Gate A estava superestimado — escrita rejeitada podia deixar
> mutação parcial e WAL envenenado. Com o fix + suíte diferencial (04dc12a)
> + identidade de ingestão por bytes e presets honestos (5d996c7), os Gates
> A–C voltam a `validated` com evidência nova. Limitações do ambiente:
> CUDA/cuVS e wheels não-Linux não validáveis aqui.

## Gate A — Fundação confiável (VALIDATED)

| Tarefa | Módulo | Teste / Evidência | Status |
|---|---|---|---|
| Tipos e modelo documental (Source/Document/Version/Chunk) | ragvault-core/types.rs | `types::tests` round-trip | validated |
| Taxonomia de erros com contexto acionável | ragvault-core/error.rs | conversões testadas via bindings | validated |
| DSL de filtros (eq/ne/in/range/contains/exists/prefix/and/or/not) | ragvault-core/filter.rs | 10 testes de semântica incl. null/NaN/depth | validated |
| Operadores desconhecidos são erro (não igualdade silenciosa) | filter.rs | `rejects_bad_shapes` | validated |
| Flat exato com filtro integrado + paralelo por shard | ragvault-vector/flat.rs | `parallel_path_matches_serial`, prefilter tests | validated |
| WAL com CRC32, truncamento de cauda corrompida, replay idempotente | ragvault-engine/wal.rs | 5 testes incl. torn tail e corrupção | validated |
| Snapshots com generations, checksums e publish atômico | ragvault-engine/snapshot.rs | testes de reopen no engine | validated |
| Recovery snapshot + WAL replay | engine.rs | `reopen_after_flush_uses_snapshot_plus_wal` | validated |
| Lock de writer único por diretório | engine.rs | `second_writer_is_rejected` | validated |
| Upsert/replace/delete atômicos por documento | engine.rs | `replace_publishes_atomically...` | validated |
| **Prepared-write: validação integral pré-WAL, apply infalível, replay de batch corrompido falha claro** | engine.rs (5aec3b9) | `rejected_write_leaves_no_trace_even_after_reopen`, `rejected_replace_preserves_old_version` | validated |
| Equivalência diferencial compact == compact+reopen (todos os backends) | tests/differential_consistency.rs (04dc12a) | 6 configs, workload misto, 2 tenants | validated |
| Identidade de fonte por sha256(raw_bytes) + fingerprints de pipeline | kb.py::sync (5d996c7) | `TestSourceIdentity` (binários distintos, invalidação por pipeline) | validated |
| Preset `quality` sem degradação lexical silenciosa; `offline-lite` explícito | kb.py/config.py (5d996c7) | `TestPresetHonesty` | validated |
| Bindings Python com exceções específicas, sem panics propagados | ragvault-python | suíte pytest | validated |

## Gate B — Retrieval competitivo (VALIDATED, com limitações registradas)

| Tarefa | Módulo | Evidência | Status |
|---|---|---|---|
| HNSW (níveis, heurística de diversidade, inserção incremental) | hnsw.rs | invariantes + recall ≥0.9 vs Flat em teste | validated |
| Recall medido em benchmark real | benchmarks/ | benchmarks/RESULTS.md | validated |
| BM25 incremental com estatísticas de docs vivos | bm25.rs | 7 testes | validated |
| Sparse vectors fornecidos pelo usuário | sparse.rs | 3 testes + engine test | validated |
| Fusão híbrida RRF ponderada | fusion.rs | testes de determinismo/pesos | validated |
| Filtro integrado à travessia HNSW + retry ef + fallback Flat exato | engine.rs::search | `filters_apply_to_all_signals` | validated |
| Planner explicável (flat vs hnsw, razões, custos) | engine.rs | plan JSON em toda busca; testes explain | validated |
| Índices tipados de metadados (bitmaps/ranges) | — | filtro avalia predicado por candidato (correto, não indexado) | deferred |
| Segmentos imutáveis + mutable segment | — | arena única + tombstones + compaction síncrona | deferred |
| Compactação | engine.rs::compact | `compact_drops_tombstones_and_preserves_results` | validated (síncrona; background deferred) |
| Concorrência: leitores paralelos + GIL liberado | bindings + testes | `test_gil_released_during_search`, `test_concurrent_reads_are_safe` | validated |

## Gate C — RAG-native (VALIDATED)

| Tarefa | Evidência | Status |
|---|---|---|
| KnowledgeBase com open/add/sync/retrieve/ask/evaluate | pytest end-to-end | validated |
| sync idempotente com hashes, include/exclude, delete_missing | `test_full_lifecycle` | validated |
| Parsers txt/md/html/json/jsonl/csv/code (+pdf/docx extras) | `TestParsers` | validated |
| Chunking estrutural (markdown sections, offsets, vizinhos) | `TestChunking` | validated |
| Embeddings plugáveis + cache sqlite | `TestEmbeddingCache` | validated |
| Agrupamento por documento + dedup + diversidade MMR | `test_duplicate_content_is_deduplicated` | validated |
| Expansão parent/child por vizinhos (não cruza documentos) | `test_context_window_expansion` | validated |
| Token budget com truncagem de fallback | `test_token_budget_is_respected` | validated |
| Citações estáveis ligadas a chunks reais | `test_full_lifecycle` (citações verificam get_chunk) | validated |
| explain + trace | `test_explain_and_trace` | validated |
| Avaliação nativa (recall/mrr/ndcg/latência) | `test_full_lifecycle` | validated |
| Presets quality/balanced/fast/offline/multilingual/code/long_documents/high_recall/low_memory | `TestConfig` | validated |
| Multi-tenancy com isolamento adversarial | `test_tenant_isolation` | validated |
| CLI init/sync/query/inspect/doctor/evaluate/compact | exercitada manualmente + smoke no CI | validated |
| Reranking plugável (callback, tolerante a falha) | `test_rerank_callback_and_tolerant_failure` | validated |
| kb.compare / kb.tune / kb.apply | `TestCompare`, `TestTune` (grid com evidência, restrição de p95, nunca auto-aplica) | validated |
| Studio UI (`ragvault studio`, stdlib http.server, local-only) | `TestStudio` | validated |

## Avaliação RAG com texto real (P1)

| Tarefa | Status | Evidência |
|---|---|---|
| Dataset real reproduzível commitado (30 passagens / 24 queries, 12 paráfrase + 12 keyword) | validated | benchmarks/data/*.jsonl |
| Harness bm25 / lexical / hybrid / +MMR / +expansion com Recall@k, MRR, nDCG, precision, dup, tokens, p50/p95 e MRR por estilo | validated | benchmarks/bench_rag_quality.py → RESULTS-RAG.md (números executados) |
| Linhas semânticas (dense/hybrid/rerank com sentence-transformers) | blocked | política de rede nega huggingface.co (CONNECT 403, registrado no proxy); comando exato documentado no harness e em RESULTS-RAG.md |

## Gate D — Performance avançada

| Tarefa | Status | Nota |
|---|---|---|
| Kernels auto-vetorizados portáteis + testes diferenciais | validated | unrolled 4-acc; sem intrinsics por arch (deferred) |
| Top-k limitado com merge por shard | validated | |
| Comparação com faiss-cpu no mesmo hardware | validated | benchmarks/RESULTS.md (cenário único; sem claim universal) |
| SQ8 (int8 + rescore f32, backend sq8_flat) | validated | testes Rust+Python, benchmark real em RESULTS.md |
| IVF-Flat (k-means determinístico, nprobe, delta scan de escritas novas) | validated | `ivf::tests` (full-probe == exato, recall monotônico em nprobe), `ivf_backend_full_lifecycle`, `TestIvfPython`; benchmark em RESULTS.md |
| IVF-PQ (ADC 8-bit, oversample 8x, rescore f32, pq_m automático) | validated | `pq_with_rescore_reaches_high_recall`, `ivf_pq_auto_subspaces` |
| OPQ / binary quantization | not-started | mesmo gate de benchmark de SQ8/IVF |
| mmap (`storage="mmap"`: base mmap + cauda RAM, checksum na abertura) | validated | `mmap_storage_full_lifecycle` (Rust), `TestMmapPython` (paridade byte-a-byte com memory) |
| Índices tipados de metadados (posting lists keyword/bool + BTreeMap numérico para ranges, interseção AND, cobertura parcial com predicado residual) | validated | `typed_prefilter_matches_predicate_results` (equivalência com predicado em 8 formas × 3 modos, deletes, compact), `typed_prefilter_is_visible_in_plan`; benchmark RESULTS-FILTERS.md: 12x @10%, 172x @1%, 484x @0.1% seletividade |
| Roaring bitmaps / histogramas / índices textuais dedicados | deferred | posting lists ordenadas cobrem eq/range; roaring é otimização de memória futura |
| Autotuning (kb.tune) | validated | grid retrieval-time com evidência por trial |
| Property testing (proptest) | validated | `proptest_filter.rs` (parser nunca em pânico; not = complemento; eq == in([x])), `proptest_topk.rs` (equivalência com sort; merge == stream único) |

## Gate E — GPU

| Tarefa | Status | Nota |
|---|---|---|
| CAGRA sidecar via cuVS (`ragvault.gpu.CagraDenseSearcher`) | implemented-experimental | plumbing testado com cuVS falso (`TestGpuPlumbing`: wiring, pós-filtro via DSL nativa, fallback CPU automático em falha); **não validado em hardware real** — runbook passo a passo em docs/GPU.md; teste real pronto (`pytest -m gpu`) para runner CUDA |
| `kb.export_dense()` (vetores para builds externos) | validated | usado por GPU sidecar e interop Faiss |
| Pós-filtro com DSL nativa (`filter_chunks`) | validated | coberto no plumbing GPU |
| Flat/IVF GPU, multi-GPU, DLPack/CUDA array interface, build GPU + serving CPU | not-started | plano e critérios de aceitação em docs/GPU.md |

## Outras pendências registradas

- ~~Sparse não persistido no WAL / compact não preserva sparse~~ — resolvido: sparse vai ao WAL (campo opcional, retrocompatível) e `compact()` faz remap de postings; testes `sparse_survives_wal_replay_and_compaction` (Rust) e `TestSparsePersistence` (Python).
- Snapshot serializa estado como JSON (formato v1) — formato binário de segmentos planejado atrás do mesmo manifest (vectors.bin já é binário e mmap-able).
- MaxSim (late interaction) — validated como estágio de reranking (`ragvault.maxsim_reranker`, `TestMaxSim`); armazenamento nativo de multivectors/named vectors — not-started.
- Interop Faiss (`ragvault.compat.faiss`, nível *convertible*) — validated com faiss-cpu real (`TestFaissCompat`: export → reconstruct → import com paridade de ranking).
- `Database`/coleções (`ragvault.Database.open`) — validated (`TestDatabase`: isolamento entre coleções, nomes validados contra path traversal, descoberta em reopen). `ragvault.connect()` — assinatura reservada com NotImplementedError explícito (remoto é pós-v0.1, ADR 0001).
- CLI `benchmark` e `migrate` — implemented (benchmark mede nesta máquina e declara isso; migrate delega à migração blocking testada).
- Integração LangChain — validated (`TestLangChain`). LlamaIndex/Haystack/DSPy — implemented (não testados contra as libs reais; erro acionável sem a dependência).
- Wheels multiplataforma: CI cobre Linux x86-64; macOS/Windows/ARM64 exigem runners não disponíveis neste ambiente.
- `kb.migrate_embeddings` — validated (estratégia blocking com swap atômico e preservação do vault antigo em falha; `TestMigrateEmbeddings`). Estratégias background/copy-on-write — planned.
