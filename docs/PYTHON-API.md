# API Python

## Módulo

```python
ragvault.open(path, preset="balanced", embedding=None, **overrides) -> KnowledgeBase
```

Presets: `quality, balanced, fast, offline, multilingual, code, long_documents, high_recall, low_memory`.

## KnowledgeBase

| Método | Descrição |
|---|---|
| `add(texts_or_dicts, metadata=None, ids=None)` | adiciona/substitui documentos; retorna ids |
| `add_documents(list_of_dicts)` | alias em lote |
| `sync(dir, include=None, exclude=None, delete_missing=True, on_error="continue")` | espelhamento idempotente → `SyncReport` |
| `async_sync(...)` | versão async (thread-offload) |
| `remove(document_id)` | exclusão (invisível em todos os índices) |
| `retrieve(query, k=8, token_budget=None, filters=None, mode=None, candidates=None, ef_search=None, rerank=None, context_window=None, max_chunks_per_document=None, explain=False, trace=False)` | → `RetrievalResult` |
| `retrieve_many(queries, **kw)` / `aretrieve` / `aretrieve_many` | lote e async |
| `ask(question, llm=..., citations=True, system_prompt=None, **retrieve_kw)` | → `Answer` (LLM é seu) |
| `evaluate(dataset, k=10)` | → `EvaluationReport` |
| `compare(dataset, presets=[...], k=10)` | avalia presets (parâmetros de retrieval) → `ComparisonReport` |
| `tune(dataset, objective="ndcg@10", max_p95_ms=None, grid=None)` | grid-search com evidência → `TuningRecommendation` (nunca aplica sozinho) |
| `apply(recommendation)` | aplica e persiste uma recomendação explicitamente |
| `as_langchain_retriever(k=8)` | retriever LangChain (requer langchain-core) |
| `as_llamaindex_retriever(k=8)` | retriever LlamaIndex (requer llama-index-core) |
| `as_haystack_retriever(k=8)` / `as_dspy_retriever(k=8)` | adaptadores Haystack 2.x / DSPy (deps opcionais) |
| `migrate_embeddings(new_embedding, strategy="blocking")` | re-embeda tudo e troca o vault atomicamente |
| `export_dense()` | (chunk_ids, float32 [n, dim]) dos vetores vivos |
| `retrieve(..., dense_searcher=s)` | candidatos densos de um sidecar (ex.: GPU CAGRA); fallback CPU automático |
| `ragvault.Database.open(root)` / `.collection(name)` | múltiplas coleções sob um diretório |
| `ragvault.maxsim_reranker(encoder)` | reranking MaxSim (late interaction) |
| `ragvault.compat.faiss` | export/import de vetores de/para Faiss (convertible) |
| `ragvault.connect(url)` | reservado para backend remoto (NotImplementedError em v0.1) |
| `for_tenant(tenant_id)` | view com isolamento automático de tenant |
| `stats()` / `inspect(document_id)` / `documents()` | introspecção |
| `flush()` / `compact()` / `close()` | durabilidade e manutenção |
| `config.explain()` / `config.export(path)` | configuração inspecionável |

Context manager: `with ragvault.open(...) as kb: ...`

## Exceções

`RagVaultError` (base Python), `ConfigurationError`, `EmbeddingError`, `IngestionError`, `EvaluationError`; nativas: `VaultError`, `VaultLockedError`, `VaultCorruptError`, `DimensionMismatchError`.
