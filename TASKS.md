# TASKS

Rastreabilidade de requisitos → implementação → evidência. Estados:
`not-started | in-progress | blocked | implemented | validated | experimental | deferred`

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
| Studio UI | — | not-started |

## Gate D — Performance avançada

| Tarefa | Status | Nota |
|---|---|---|
| Kernels auto-vetorizados portáteis + testes diferenciais | validated | unrolled 4-acc; sem intrinsics por arch (deferred) |
| Top-k limitado com merge por shard | validated | |
| Comparação com faiss-cpu no mesmo hardware | validated | benchmarks/RESULTS.md (cenário único; sem claim universal) |
| SQ8 / IVF / PQ / OPQ / binary | not-started | |
| mmap / storage híbrido | not-started | |
| Autotuning (kb.tune) | validated | grid retrieval-time com evidência por trial |

## Gate E — GPU

| Tarefa | Status | Nota |
|---|---|---|
| Flat GPU / CAGRA / IVF GPU via cuVS | blocked | ambiente sem GPU; nenhuma integração foi iniciada para não criar stubs enganosos |
| DLPack / CUDA array interface | not-started | |

## Outras pendências registradas

- ~~Sparse não persistido no WAL / compact não preserva sparse~~ — resolvido: sparse vai ao WAL (campo opcional, retrocompatível) e `compact()` faz remap de postings; testes `sparse_survives_wal_replay_and_compaction` (Rust) e `TestSparsePersistence` (Python).
- Snapshot serializa estado como JSON (formato v1) — formato binário de segmentos planejado atrás do mesmo manifest.
- Named vectors / multivectors (MaxSim) — not-started.
- Integração LangChain — validated (`TestLangChain`, adapter com proveniência completa). LlamaIndex — implemented (não testado: dependência não instalada no ambiente; erro acionável). Haystack/DSPy — not-started.
- Wheels multiplataforma: CI cobre Linux x86-64; macOS/Windows/ARM64 exigem runners não disponíveis neste ambiente.
- `kb.migrate_embeddings` — not-started (reopen com embedder diferente é rejeitado com erro claro).
