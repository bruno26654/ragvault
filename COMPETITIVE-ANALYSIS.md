# Análise competitiva (scorecard honesto, jul/2026)

| Capacidade | RagVault 0.1 | Faiss | Qdrant (local) | LanceDB | Chroma |
|---|---|---|---|---|---|
| Documentos → contexto citável em uma API | ✅ | ❌ (só índices) | ❌ (só vetores+payload) | parcial | parcial |
| Ingestão idempotente por hash (`sync`) | ✅ | ❌ | ❌ | ❌ | ❌ |
| Híbrido dense+BM25 nativo com fusão | ✅ | ❌ | ✅ | ✅ (tantivy) | parcial |
| Filtro integrado à travessia ANN | ✅ (traversal-aware + fallback exato) | ❌ | ✅ | ✅ | pós-filtro |
| WAL + crash recovery embedded | ✅ | ❌ | ✅ (server) | ✅ | parcial |
| Citações com documento+versão | ✅ | ❌ | ❌ | ❌ | ❌ |
| Avaliação nativa de retrieval | ✅ | ❌ | ❌ | ❌ | ❌ |
| Velocidade bruta ANN CPU | faiss ≈3× mais rápido no cenário medido (benchmarks/RESULTS.md) | ✅ referência | — | — | — |
| Quantização (SQ/PQ/IVF) | ❌ planejado | ✅ | ✅ | ✅ | ❌ |
| GPU | ❌ planejado | ✅ | parcial | ❌ | ❌ |
| Maturidade/ecossistema | alpha | ✅ | ✅ | ✅ | ✅ |

Posição: RagVault não compete em velocidade bruta de ANN hoje (medido e admitido); compete em **fluxo RAG completo com durabilidade e proveniência embutidas**. Metas de performance ficam nos Gates D/E de TASKS.md com critérios de medição definidos (ADR 0015).
