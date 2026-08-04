# TASKS

Rastreabilidade de requisitos → implementação → evidência. Estados:
`not-started | in-progress | blocked | implemented | under-review | validated | experimental | deferred`

> **Auditoria final (1.0.0-rc1, main 22e880d):** gates re-executados —
> fmt/clippy limpos; **122 testes Rust** (unit/lib + differential multi-segmento
> + proptest) e **89 Python** verdes (6 skipped: roundtrips de framework sem a
> lib local + teste `gpu`; os roundtrips rodam no job `integrations` do CI com
> versões fixadas). Matriz completa de CI verde na main: rust, py 3.9/3.11/3.12,
> integrações reais, wheels Linux x86-64/aarch64 + macOS arm64 + Windows x86-64
> com smoke de clean-install. Histórico do hardening: P0 de atomicidade
> reproduzido e corrigido (5aec3b9), suíte diferencial (04dc12a), identidade
> por raw-bytes + presets honestos (5d996c7), context v2 (dd1ce16), eval texto
> real (ae8fe9f), filtros tipados (3b7fddf), batch nativo (cd95482), HNSW
> `VisitedSet` geracional comprovado (RESULTS-HNSW.md), correções Windows,
> **storage v2 completo (ADR 0016)** e compactação read-friendly. Gates A–D
> `validated`. **Nenhum P0/P1/P2 arquitetural aberto.** Pendências externas:
> GPU real (sem hardware). O eval semântico foi executado externamente e está
> registrado com números medidos em benchmarks/RESULTS-RAG.md.

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
| Índices tipados de metadados (bitmaps/ranges) | engine.rs (3b7fddf) | ver Gate D (posting lists keyword/bool + BTreeMap numérico, plano, RESULTS-FILTERS.md) | validated |
| Segmentos imutáveis + mutable segment | — | hoje: arena única + tombstones + compaction síncrona; storage v2 segmentado é a próxima tarefa (ver rodapé) | deferred |
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
| Linhas semânticas (dense/hybrid/rerank com sentence-transformers) | validated | executadas em máquina com acesso ao modelo (2026-08-02): **semantic dense recall@5 1.000, MRR 0.979, MRR-paráfrase 0.958** vs 0.417 do baseline lexical. Dois achados registrados: fusão híbrida *piora* o dense semântico (MRR 0.979 → 0.785) e o rerank custa 40× de latência (10.1 → 411.9 ms p50) sem ganho algum de qualidade neste corpus. RESULTS-RAG.md |

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
- Integrações LangChain/LlamaIndex/Haystack/DSPy — validated (`TestIntegrations`): roundtrips reais contra versões fixadas no job `integrations` do CI; erro acionável quando a lib está ausente. Localmente os roundtrips fazem `importorskip`.
- Wheels multiplataforma — implemented: matriz de CI (`.github/workflows/ci.yml` job `wheels`) constrói Linux x86-64/aarch64, macOS arm64 (macos-14) e Windows x86-64, cada um com smoke de clean-install (`open → add → retrieve`). Execução dos runners não-Linux ocorre no CI, não neste ambiente. Changelog/checklist de release em `CHANGELOG.md` e `docs/RELEASE.md`.
- `kb.migrate_embeddings` — validated (estratégia blocking com swap atômico e preservação do vault antigo em falha; `TestMigrateEmbeddings`). Estratégias background/copy-on-write — planned.

## Pipeline multi-query (perguntas compostas)

Motivado por falhas reais de uso: perda de recall em perguntas compostas,
entrada de distratores, citações inconsistentes e rerank caro em CPU.

| Tarefa | Status | Evidência |
|---|---|---|
| `retrieve_multi()` / `ask_multi()` (+ `aretrieve_multi`/`aask_multi`) | validated | `TestMultiHopRecall`, `TestAskMulti`; API pública exportada (`MultiRetrievalResult`) |
| Decomposição por callback externo + subconsultas manuais + fallback seguro | validated | `TestDecomposerFallback` (exceção, `None` e lixo → consulta única, registrado no trace) |
| Execução em lote via `search_many` (uma chamada nativa, GIL liberado) | validated | `TestCpuPerformance::test_multi_query_stays_within_a_small_factor_of_single` (6 consultas < 6× uma) |
| Fusão global Weighted RRF + dedup por `chunk_id` | validated | `trace.fusion` com contribuição por ranking; determinístico (desempate por chunk_id) |
| **Garantia de cobertura por subconsulta** | validated | Defeito real encontrado por benchmark: pool continha 24/24 evidências mas o contexto final só 5/24 — RRF (k0=60) achata rank 1 vs 10 em ~16%, então distratores de consenso raso enterravam a evidência especialista. Tier de reserva corrige: full-recall 0.167 → 0.875 (`TestCoverageGuarantee`, RESULTS-MULTIQUERY.md) |
| Precedência documental por metadados (status/data/versão/tipo) | validated | `TestVersionPrecedence`; conflitos explícitos em `result.conflicts` + trace + prompt |
| Filtros obrigatórios (prefilter nativo) e boosts semânticos pós-fusão | validated | `test_mandatory_filter_excludes_revoked_before_search`, `test_boost_prefers_document_type` |
| Rerank global que não destrói recall + tolerante a falha | validated | `TestRerankSafety` (reranker adversarial que descarta tudo → candidatos recuperados; exceção → ordem fundida mantida) |
| MMR + Context Builder após a fusão, orçamento **global** de tokens | validated | `test_global_token_budget_is_respected` |
| Expansão de vizinhos só após a seleção final (distratores não expandem) | validated | `test_distractors_do_not_get_neighbor_expansion` |
| Proveniência e citações: só documentos recuperados, sem `[n]` inventado | validated | `test_only_retrieved_documents_are_cited`, `test_answer_keeps_only_real_citations`, `test_prompt_forbids_question_facts_as_evidence` |
| Trace completo (subconsultas, candidatos, contribuições, fusão, eliminados+motivo, scores pré/pós rerank, filtrados, tempo por etapa) | validated | `TestTraceCompleteness` |
| Compatibilidade com `retrieve()`/`retrieve_many()`/`ask()` | validated | `TestCompatibility` (comportamento single-query inalterado) |
| Checklist de facetas no prompt de geração (subconsultas cobrem recuperação **e** resposta) | validated | `TestAnswerFacetChecklist` (facetas listadas, pergunta original não repetida, ausentes quando há só uma consulta ou a decomposição falha, `system_prompt` customizado preservado, ordem contexto→facetas→pergunta, `trace["answer_facets"]`). Inclui saída explícita para faceta sem evidência — evita empurrar o modelo a inventar. Efeito na geração não medido aqui (exige LLM real) |
| Benchmark single vs multi (precisão, recall, latência, tokens) | validated | `benchmarks/bench_multiquery.py` + dataset multi-hop commitado (24 perguntas: 20 two-hop + 4 three-hop) → RESULTS-MULTIQUERY.md |
| Exemplo funcional com Groq (sem dependência obrigatória) | validated | `examples/multi_query_rag.py` (roda offline por padrão; `--groq` opcional) |

## Validação semântica pós-geração (`verify=`)

Complementa a integridade de citações: aquela barra `[n]` inventado, esta pega
a citação que **existe mas não sustenta** a afirmação.

| Tarefa | Status | Evidência |
|---|---|---|
| Verificador opcional por callback em `ask()` e `ask_multi()` | validated | `TestCompatibility` (sem `verify=` o comportamento é idêntico ao anterior) |
| Extração de afirmações + citações `[n]` associadas + resolução para chunks reais | validated | `test_evidence_resolves_to_real_chunks`, `TestClaimSplitting` |
| Vereditos `supported/unsupported/contradicted/uncited/question_fact/inference` | validated | `TestWrongButExistingCitation`, `TestQuestionFactVsDocumentEvidence` |
| Distinção fato-da-pergunta × evidência documental × inferência | validated | `test_question_fact_is_distinguished_from_evidence`, `test_inference_is_its_own_verdict` |
| Modos `report` / `annotate` / `repair` / `strict` | validated | um teste por modo; `repair` **remove** o que não se sustenta |
| Conflito vigente × revogado detectado na resposta | validated | `TestVersionConflict` (afirmação com regra revogada é reparada) |
| Afirmação sem citação | validated | `TestUncitedClaims` (`strict` remove, `repair` mantém) |
| Falha do verificador preserva a resposta válida | validated | `TestVerifierFailure` (exceção, `None`, contagem errada de vereditos → resposta intacta + erro registrado) |
| Veredito/modo inválido são erro acionável (não "ok" silencioso) | validated | `test_unknown_verdict_is_an_actionable_error`, `test_unknown_mode_is_rejected` |
| Trace: afirmação, citações, status, justificativa, correções e tempo | validated | `TestTraceAndReport` |
| **Falha fechada**: `ok`/`complete` só `True` com verificação completa e estruturalmente válida | validated | `TestFailsClosed` — bug real: verificador quebrado dava `ok=True` (`not any([])`), e cobertura parcial dava `complete=True` |
| Validade estrutural como quarto eixo (`valid`, `structural_issues`) | validated | `TestSegmentationStructure` (sobreposição/ordem recusadas, texto não coberto sinalizado) |
| `replacement` aceito só com veredito `supported` na revalidação | validated | `TestReplacementMustBeSupported` (uncited/inference/question_fact não são endosso) |
| Segmentação: CJK/árabe/hebraico/índico/etíope/armênio/khmer + guarda de abreviações + segmentação opcional pelo verificador | validated | `TestClaimBoundaries` (13 scripts/casos, split lossless), `TestVerifierSegmentation` (divide duas afirmações numa frase; paráfrase recusada). Bugs reais: o lookahead exigia maiúscula, então idiomas sem caixa **nunca** eram divididos; e o danda `।` faltava, deixando híndi/bengali/marathi/nepali como uma única claim |
| Pontuação do mundo real: espaço ausente, terminador ausente, iniciais de nome | validated | `test_malformed_punctuation_still_splits` / `test_does_not_invent_boundaries`. Bug real: "John F. Kennedy" era dividido na inicial. Minúscula após ponto **não** divide de propósito — abreviatura minúscula é conjunto aberto e a divisão falsa vira fragmento que o `repair` apaga |
| Segmentador conectável (`segmenter=`) para prosa sem terminador (tailandês/laosiano/birmanês) | validated | `TestSuppliedSegmenter` (claims não-verbatim e sobrepostas recusadas, texto não coberto sinalizado, segmentador quebrado preserva a resposta). Sem dependência no núcleo — callable do chamador, como `verify=`/`rerank=` |
| Especificidade da `quote` (`max_quote_occurrences`) | validated | `TestQuoteSpecificity`. Bug real: `quote="o"` passava no teste de substring. Contagem de ocorrências em vez de tamanho — independente de script; contada por fonte, então chunk duplicado não pune citação legítima |
| Verificador NLI offline (`ragvault.nli`) | **validated** (medido; restrito a `report`/`annotate`) | `tests/python/test_nli.py` (33 testes com modelo stub: mapeamento por nome sob `id2label` embaralhado, granularidade de premissa, agregação, `uncited` sem chamada ao modelo). Medido em `RESULTS-VERIFICATION.md` (36 pares, en/es/pt, mDeBERTa-v3-base-xnli, CPU): 0.89 de acurácia com premissa nua, **0.78 com premissa realista e 21% de falso-`contradicted`** — uma afirmação correta em cinco seria apagada pelo `repair`. Portão reprovado → o adaptador **recusa `repair`/`strict` por padrão** (`TestDestructiveModeGuard`), `allow_repair=True` libera. p50 ~10 s/afirmação (22 s com premissa realista) em CPU |
| Formatação preservada no reparo (listas/parágrafos) + bullets como fronteira de claim | validated | `TestFormattingPreserved` (bug real: o regex exigia maiúscula após o separador, então uma lista inteira era uma única claim) |
| Segunda verificação sobre os `replacement` (uma passagem, sem laço) | validated | `TestReplacementRecheck` (rejeitado → descartado; falha → reparo mantido + `recheck_error`) |
| Cobertura de facetas (completude ≠ fidelidade), sem preenchimento automático | validated | `TestFacetCoverage`; `complete` só é `None` quando não havia facetas — faceta declarada e não avaliada falha fechada (`test_unreported_facet_counts_as_uncovered`) |
| O verificador segmenta e classifica, não escreve (`replacement` ignorado por padrão) | validated | `test_replacements_are_ignored_by_default`; revalidar o próprio `replacement` é autoendosso. `allow_replacements=True` restaura o comportamento anterior (`test_repair_uses_a_replacement_when_opted_in`) |
| Critério de aceite: `ok`/`complete` só `True` com todas as claims sustentadas e todas as facetas cobertas | validated | `TestSemanticHardening` — contradição com a pergunta, claim histórica sem fonte histórica, faceta composta parcial, proposição composta resegmentada, retorno incompleto, e a matriz `ok × complete` |
| Metadados de precedência na evidência e em `Citation.metadata` | validated | `TestEvidenceMetadata` |
| Texto do bloco citado na evidência (`Citation.text`) | validated | `TestCitedEvidenceIsAtHand` — sem ele o juiz reencontra `[n]` no contexto e pode justificar a claim com um bloco que ela não citou |
| Todo veredito nomeia um lastro conferível (`require_evidence`) | validated | `TestEveryVerdictNamesItsGround` — bug real: `supported`/`inference`/`question_fact` passavam com `ok=True` sem citar nem citar trecho, então bastava rotular uma frase inventada. `supported` sem citação vira `uncited`; `ask`/`ask_multi` derivam o flag do próprio `citations` |
| `quote` conferido contra o lastro do veredito (fonte × pergunta) | validated | `TestEveryVerdictNamesItsGround::test_question_fact_is_checked_against_the_question` — bug real: `question_fact` citando a pergunta era comparado aos documentos e reprovado |
| `quote` conferido contra a fonte citada; `require_quotes` | validated | `TestQuotedEvidence` (citação fabricada → `unsupported` + `structural_issues`; reformatação/caixa não reprovam; opcional por padrão porque suporte nem sempre é trecho contíguo) |
| Marcador de citação depois do terminador fica com a claim que ele fonteia | validated | `TestTrailingCitationMarkers` — bug real: `"30 dias. [1]"` deixava a claim `uncited` (apagada por `strict`), creditava a próxima e criava uma "claim" que era só `[2]`; sem espaço não dividia |
| `facets=` explícito em `ask()`/`ask_multi()` (obrigação ≠ subconsulta de busca) | validated | `TestExplicitFacets` — subconsulta de recuperação virava obrigação artificial; facetas declaradas valem para o checklist do prompt e para o julgamento |
| Filtros por subconsulta (`subquery_filters`) | validated | `TestPerSubqueryFilters` (faceta decisória em `VIGENTE` + histórica em `REVOGADO` na mesma chamada) |
| Cenário de alta exigência: base grande, ruidosa e versionada | validated | `test_scenario_versioned_registry.py` — ~900 docs de ruído + regras vigentes/revogadas/redundantes/incompletas/conflitantes em 4 entidades; achou os três bugs de resolução de versão abaixo |
| Revogação é absoluta (não só relativa ao sucessor recuperado) | validated | `TestRevocationIsAbsolute` — bug real: com o sucessor fora do pool, a regra revogada entrava no contexto parecendo vigente. Filtro explícito de status do chamador tem precedência |
| Empate de precedência não é decidido por ordem alfabética | validated | `TestUndecidableConflict` — bug real: `document_id` desempatava e a evidência do perdedor sumia; agora ambos ficam, `resolved=False`/`tied`, e o prompt manda relatar a divergência |
| Eliminação não encolhe o contexto nem gasta a vaga de cobertura | validated | `TestRevocationIsAbsolute::test_the_replacement_takes_the_freed_slot` — bug real: a vaga da faceta era gasta na versão revogada e a vigente ficava fora da janela; fusão e resolução rodam até ponto fixo |
| Sem dependência obrigatória de provedor | validated | verificador é callable; exemplo roda offline, `--groq` opcional |

## Backlog v1.0 restante (apenas performance/formato — nenhum P0/P1 aberto)

| Tarefa | Status | Critério de conclusão |
|---|---|---|
| Profiling HNSW e correção de gargalos comprovados | validated | Perfil medido (`examples/hnsw_bench.rs`): a alocação O(N) de `visited` por `search_layer` foi provada como gargalo em escala. Substituída por `VisitedSet` geracional thread-local (reset O(1), reuso por thread). Recall idêntico; em N=200k: mean 1.9×, p95 3.7×, p99 10.4×, QPS 1.9× (RESULTS-HNSW.md). Neutro em N=50k. Heaps de scratch (limitados por ef, não N) e SIMD não perseguidos — sem gargalo comprovado. |
| Storage v2 — container de segmento binário + base v2 + migração | validated | `crate::segment` (records com CRC por registro + CRC de stream no footer, verificação incremental; `segment::tests`). `format_version = 2` grava a base como `gen-N/state.rvseg` (binário) em vez de `state.json`; migração v1→v2 transparente na reabertura (`v2_flush_writes_binary_segment`, `v1_vault_migrates_to_v2_on_reopen_and_flush`, `corrupt_snapshot_is_detected_not_silently_loaded`). ADR 0016. |
| Storage v2 — deltas multi-segmento + flush O(delta) + compactação (fold) | validated | flush anexa segmento de delta binário (ops do WAL) em vez de reescrever a base; orçamento `MAX_DELTA_SEGMENTS` dispara reescrita total; `compact()` colapsa deltas em nova base; open aplica base→deltas→WAL. Testes: `delta_flush_accumulates_segments_and_reopens`, `delta_tombstone_and_replace_shadow_base`, `delta_budget_exhaustion_rewrites_base`, `orphan_delta_segment_is_ignored_on_reopen`, `corrupt_delta_segment_is_detected_not_silently_loaded`, `differential_multi_segment_deltas`. ADR 0016. |
| Storage v2 — read-during-compaction sem bloqueio | validated | Rebuild da compactação roda fora do lock sobre snapshot clonado (read lock curto); write lock só para swap+publish; escrita concorrente invalida o snapshot via seq → retry off-lock e fallback sob lock (sempre correto). Testes: `readers_keep_searching_during_compaction` (4 leitores em loop durante compact, resultados exatos idênticos antes/depois), `concurrent_write_during_compaction_is_preserved` (20 escritas concorrentes preservadas + reopen). **ADR 0016 completa — nenhum item de storage v2 restante.** |
| Roaring bitmaps / histogramas / OPQ / binary quantization | deferred | Só substituir posting lists / quantizadores atuais quando profiling ou benchmark demonstrar necessidade. |
