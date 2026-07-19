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
| `for_tenant(tenant_id)` | view com isolamento automático de tenant |
| `stats()` / `inspect(document_id)` / `documents()` | introspecção |
| `flush()` / `compact()` / `close()` | durabilidade e manutenção |
| `config.explain()` / `config.export(path)` | configuração inspecionável |

Context manager: `with ragvault.open(...) as kb: ...`

## Exceções

`RagVaultError` (base Python), `ConfigurationError`, `EmbeddingError`, `IngestionError`, `EvaluationError`; nativas: `VaultError`, `VaultLockedError`, `VaultCorruptError`, `DimensionMismatchError`.
