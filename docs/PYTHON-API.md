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
| `retrieve_multi(question, subqueries=None, decompose=None, max_subqueries=6, fusion="weighted_rrf", coverage_per_subquery=1, rerank=None, filters=None, subquery_filters=None, boosts=None, resolve_versions=False, **retrieve_kw)` | pipeline multi-query → `MultiRetrievalResult` |
| `ask_multi(question, llm=..., citations=True, verify=None, verification_mode="report", **retrieve_multi_kw)` / `aretrieve_multi` / `aask_multi` | multi-query + LLM com integridade de citações |
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
| ↳ *subconsultas atômicas* | cada subconsulta deve cobrir **uma** obrigação de resposta. Uma subconsulta composta reserva 1 vaga de cobertura para 2 evidências e traz só uma; separada em duas, cada evidência ganha a sua vaga |
| `coverage_per_subquery` | quantos top-hits de cada subconsulta são reservados (0 desliga) |
| `fusion_weights=[...]` | peso por consulta (pergunta original primeiro) |
| `filters={...}` | filtro **obrigatório**, aplicado como prefilter nativo antes da busca |
| `boosts=[{"filter": {...}, "weight": 2.0}]` | boost multiplicativo **depois** da fusão |
| `resolve_versions=True` | precedência por metadados dentro de `doc_group` |
| `subquery_filters=[...]` | filtro **por consulta** (pergunta original primeiro); `None` mantém o global. Uma entrada **substitui** o filtro global daquela consulta — é o que permite "faceta decisória só em `VIGENTE`, faceta histórica só em `REVOGADO`", impossível com um filtro único |
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
| `contradicted` | a fonte citada **ou um fato explícito da pergunta** contradiz a afirmação — vale mesmo sem fonte documental envolvida |
| `uncited` | afirmação sem citação |
| `question_fact` | **apenas** repetição de fato dado na pergunta; conclusões, aplicações de regra e deduções são `inference`, não `question_fact` |
| `inference` | inferência derivada, não afirmação documental |

| Modo | Ação |
|---|---|
| `report` (padrão) | não altera nada; só anexa o relatório |
| `annotate` | marca inline `[unsupported]`/`[contradicted]` |
| `repair` | remove afirmações problemáticas |
| `strict` | como `repair` e também remove as `uncited` |

**O verificador segmenta e classifica; ele não escreve.** Um verificador que
propõe um `replacement` e depois o revalida está avaliando o próprio texto —
autoendosso, não verificação. Por isso `replacement` é **ignorado por padrão**
(`allow_replacements=False`): a afirmação que não se sustenta é **removida**,
não reescrita. Quem aceita essa troca liga `allow_replacements=True` em
`ask()`/`ask_multi()`; aí vale a segunda passagem descrita abaixo.

**Fidelidade × completude** são eixos separados: toda afirmação pode estar
sustentada e a resposta ainda deixar uma faceta de fora. Quando a pergunta foi
decomposta, as facetas vão no payload (`payload["facets"]`) e o verificador
pode devolver `{"claims": [...], "facets": [{"facet": ..., "covered": bool,
"rationale": ...}]}`. O relatório expõe `facet_coverage`, `uncovered_facets` e
`complete` — que só é `None` quando **não havia facetas a cobrir**. Declarada
uma faceta, silêncio sobre ela não é desconhecido: é faceta **não avaliada**, e
falha fechada como um relatório parcial. **Faceta composta** só é `covered=true`
quando *todos* os seus componentes forem respondidos — julgamento que cabe ao
verificador (está no prompt dos exemplos), não à biblioteca. Faceta descoberta é
**reportada, nunca preenchida automaticamente**: regenerar exigiria uma
chamada extra ao LLM que o chamador não pediu, com custo e risco de laço —
a decisão fica com quem chama.

**Segunda passagem sobre os `replacement`** (só com `allow_replacements=True`):
em `repair`/`strict`, o texto proposto pelo verificador é ele próprio
verificado uma vez (nunca em laço) — pelo mesmo verificador, o que é
exatamente a razão de o padrão ser não aceitar reescrita.
Replacement reprovado é descartado em vez de substituído de novo, e
`claim.replacement_verdict` registra o veredito. Se essa segunda passagem
falhar, o reparo é mantido e `recheck_error` declara que os replacements não
foram checados.

**Formatação preservada**: `repair` reconstrói a resposta com os separadores
originais — listas e parágrafos sobrevivem à remoção de um item.

**Segmentação das afirmações**: a divisão embutida é heurística — terminadores
de sentença, marcadores de lista, pontuação **CJK/árabe/hebraica** e guarda de
abreviações (`Art.`, `Inc.`, `Dr.`…). Antes o lookahead exigia maiúscula, então
respostas em chinês, japonês, árabe ou hebraico **nunca eram divididas** e a
verificação por afirmação era inócua nesses idiomas.

O heurístico não enxerga *duas* afirmações dentro de uma frase
(`"X leva 30 dias [1] e Y leva 5 dias [2]"`) — e nesse caso reprovar `[2]`
apagaria também a parte correta. Para isso o verificador pode devolver a
**própria segmentação**: itens com a chave `claim`, que precisam ser
substrings **verbatim** da resposta. Custa zero chamadas extras (o verificador
já é um LLM lendo a resposta inteira) e mantém o reparo cirúrgico — remove-se
o trecho do original em vez de reescrever a partir de texto do modelo. Uma
segmentação que parafraseie é recusada, e `report.segmentation` diz qual foi
usada (`"heuristic"` ou `"verifier"`).

**Metadados na evidência**: cada `evidence` traz o `metadata` efetivo do
documento citado (incluindo `status`, data de vigência e versão), também
disponível em `Citation.metadata` — sem isso o verificador não consegue
distinguir uma regra vigente de uma revogada.

### Quatro eixos independentes

| Eixo | Pergunta que responde |
|---|---|
| `ok` | as afirmações se sustentam? (fidelidade) |
| `complete` | todas as facetas foram cobertas? (completude) |
| `valid` | a saída do verificador é estruturalmente sã? |
| `segmentation` | quem dividiu as afirmações? |

**Falha fechada.** `ok` e `complete` só podem ser `True` quando a verificação
**completou** e o resultado é estruturalmente válido:

- verificador que levantou exceção → `ok=False` (antes `not any([])` dava
  `True`: um verificador quebrado reportava a resposta como fiel, e quem
  fizesse `if verification.ok:` embarcava texto não verificado);
- faceta esperada que o verificador não mencionou → conta como **não coberta**
  e aparece em `uncovered_facets`; se ele não mencionou **nenhuma**, `complete`
  é `False`, não `None` — facetas foram declaradas e nada mostrou que foram
  cobertas;
- `complete=None` significa *desconhecido*, e o único desconhecido honesto é
  "não havia facetas a cobrir" — nunca é um `True` silencioso;
- fato dado na pergunta não é afirmação sem fonte: `question_fact` **não** é
  `uncited`, e nem mesmo `strict` o remove — apagá-lo seria descartar um
  enunciado verdadeiro por faltar um documento que não pode existir;
- trecho da resposta que nenhuma afirmação cobriu → `structural_issues`
  registra quantos caracteres ficaram **sem verificação**, e `ok` cai.

**Requisitos estruturais da segmentação do verificador** (checáveis sem
julgar significado): substrings literais, ordem preservada e **sem
sobreposição** — spans sobrepostos julgariam o mesmo texto duas vezes e fariam
o reparo produzir lixo. Violação é recusada e a resposta original preservada.

**Replacement só entra se `supported`** — e só com `allow_replacements=True`.
O texto do `replacement` foi escrito pelo verificador, não pelo modelo que o
usuário escolheu: `uncited`, `inference` ou `question_fact` na revalidação
**não são endosso** e o trecho é removido em vez de substituído.

**Critério de aceite:** `ok=True` e `complete=True` só quando *todas* as
afirmações se sustentam e *todas* as facetas foram integralmente cobertas.
São eixos independentes — uma resposta fiel pode ser incompleta e vice-versa —
e ambos falham fechado.

**Garantias:** verificador que levanta exceção, devolve `None` ou um número
errado de vereditos **preserva a resposta original** e registra o erro
(`answer.verification.error`) — um verificador quebrado nunca destrói uma
resposta válida, e também nunca a declara fiel. Veredito ou modo desconhecido são `ConfigurationError`
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
