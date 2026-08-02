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
| `retrieve_multi(question, subqueries=None, decompose=None, max_subqueries=6, fusion="weighted_rrf", coverage_per_subquery=1, rerank=None, filters=None, boosts=None, resolve_versions=False, **retrieve_kw)` | pipeline multi-query → `MultiRetrievalResult` |
| `ask_multi(question, llm=..., citations=True, **retrieve_multi_kw)` / `aretrieve_multi` / `aask_multi` | multi-query + LLM com integridade de citações |
| `ask(question, llm=..., citations=True, system_prompt=None, verify=None, verification_mode="report", **retrieve_kw)` | → `Answer` (LLM é seu) |
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

## Multi-query (`retrieve_multi` / `ask_multi`)

Para perguntas compostas, onde uma consulta única perde recall porque cada
faceta compete pelo mesmo espaço de candidatos.

```python
answer = kb.ask_multi(
    question,
    llm=answer_llm,              # callable(prompt) -> texto (seu provedor)
    decompose=query_decomposer,  # callable(pergunta) -> [subconsultas]
    max_subqueries=6,
    fusion="weighted_rrf",
    rerank=reranker,
    filters={"status": "VIGENTE"},
    resolve_versions=True,
    citations=True,
    explain=True,
    trace=True,
)
```

Etapas: decomposição (opcional) → busca em lote nativa (`search_many`, GIL
liberado) → **Weighted RRF global** com dedup por `chunk_id` → **garantia de
cobertura por subconsulta** → precedência de versões → boosts → rerank global
→ MMR + Context Builder com orçamento **global** de tokens → expansão de
vizinhos só para a seleção final.

**Por que a garantia de cobertura existe.** Só RRF não basta: com `k0=60`, a
diferença entre rank 1 e rank 10 é ~16%, então um documento medíocre que
aparece no meio do ranking de *todas* as subconsultas acumula mais massa que o
documento especialista que é o top-1 de exatamente uma faceta — e a evidência
daquela faceta some. `coverage_per_subquery` (padrão 1) reserva os melhores
resultados de cada subconsulta em um tier prioritário. Medido em
`benchmarks/RESULTS-MULTIQUERY.md`: full-recall 0.167 → 0.875.

| Parâmetro | Efeito |
|---|---|
| `subqueries=[...]` | subconsultas manuais (ignora `decompose`) |
| `decompose=fn` | callback externo; **qualquer falha cai para consulta única** |
| `coverage_per_subquery` | quantos top-hits de cada subconsulta são reservados (0 desliga) |
| `fusion_weights=[...]` | peso por consulta (pergunta original primeiro) |
| `filters={...}` | filtro **obrigatório**, aplicado como prefilter nativo antes da busca |
| `boosts=[{"filter": {...}, "weight": 2.0}]` | boost multiplicativo **depois** da fusão |
| `resolve_versions=True` | precedência por metadados dentro de `doc_group` |
| `rerank=fn` | rerank global; **nunca destrói recall** (descartados voltam) e falha tolerada |

**Precedência de versões** (`resolve_versions=True`): dentro de cada
`doc_group`, ordena por status (`VIGENTE`/`active` > desconhecido >
`REVOGADO`/`revoked`), depois `effective_date` mais recente, depois `version`
maior. Os perdedores **nunca** são silenciosos: aparecem em
`result.conflicts` e no trace, e `ask_multi` os declara no prompt.

**Checklist de facetas na resposta**: as subconsultas não servem só à
recuperação — `ask_multi` também as lista no prompt como facetas que a
resposta deve cobrir. Sem isso, o modelo pode receber toda a evidência e
ainda responder só uma faceta. O prompt inclui uma saída explícita: faceta
sem evidência no contexto deve ser **declarada como não respondida**, nunca
inventada (as facetas enviadas ficam em `trace["answer_facets"]`).

**Integridade de citações**: só documentos recuperados podem ser citados;
marcadores `[n]` inexistentes no contexto são removidos da resposta; o prompt
declara explicitamente que fatos da pergunta não são evidência documental.

`MultiRetrievalResult` = `RetrievalResult` + `.subqueries` + `.conflicts`.

## Validação semântica pós-geração (`verify=`)

A integridade de citações barra marcador `[n]` **inventado**. O que ela não
pega é a citação que **existe mas não sustenta** a afirmação, o fato da
pergunta apresentado como evidência documental, ou a afirmação contradita
pela própria fonte citada. Para isso há um verificador opcional por callback:

```python
answer = kb.ask_multi(
    question, llm=answer_llm,
    verify=semantic_verifier,      # callable(payload) -> [veredito por afirmação]
    verification_mode="repair",
    trace=True,
)

print(answer.text)                  # já reparado
print(answer.verification.ok)       # False se algo não se sustenta
print(answer.unverified_claims)     # afirmações problemáticas
```

O verificador recebe um payload com `question`, `answer`, `context` e, por
afirmação, `claim` + `citations` + `evidence` (documento, versão e **chunk_ids
reais**). Devolve um veredito por afirmação — string ou dict com `verdict`,
`rationale` e `replacement` opcional.

| Veredito | Significado |
|---|---|
| `supported` | a fonte citada sustenta a afirmação |
| `unsupported` | a citação não sustenta o que foi afirmado |
| `contradicted` | a fonte citada **contradiz** a afirmação |
| `uncited` | afirmação sem citação |
| `question_fact` | fato vindo da pergunta, não do documento |
| `inference` | inferência derivada, não afirmação documental |

| Modo | Ação |
|---|---|
| `report` (padrão) | não altera nada; só anexa o relatório |
| `annotate` | marca inline `[unsupported]`/`[contradicted]` |
| `repair` | remove afirmações problemáticas, ou usa `replacement` |
| `strict` | como `repair` e também remove as `uncited` |

**Garantias:** verificador que levanta exceção, devolve `None` ou um número
errado de vereditos **preserva a resposta original** e registra o erro
(`answer.verification.error`) — um verificador quebrado nunca destrói uma
resposta válida. Veredito ou modo desconhecido são `ConfigurationError`
acionável, nunca tratados como "ok" em silêncio. Se tudo for removido, a
resposta declara isso em vez de ficar vazia. Nenhum provedor é obrigatório.

**Trace** (`trace=True`) em `trace["verification"]`: `mode`, `ok`, `counts`,
`elapsed_ms` e, por afirmação, `claim`, `citations`, `chunk_ids`, `verdict`,
`rationale`, `replacement` e `action` (`kept`/`annotated`/`rewritten`/`removed`).

**Trace** (`trace=True`): `subqueries`, `candidates_per_subquery`, `fusion`
(método, k0, contribuição de cada ranking por chunk, `fused_score` e
`coverage_reserved`), `version_conflicts`, `boosts`, `rerank`
(scores antes/depois, itens recuperados), `eliminated` (item + motivo),
`decompose_error`/`rerank_error` + fallback aplicado, e `stage_ms` por etapa.

## Exceções

`RagVaultError` (base Python), `ConfigurationError`, `EmbeddingError`, `IngestionError`, `EvaluationError`; nativas: `VaultError`, `VaultLockedError`, `VaultCorruptError`, `DimensionMismatchError`.
