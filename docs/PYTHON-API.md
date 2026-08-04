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
| `ask_multi(question, llm=..., citations=True, verify=None, verification_mode="report", segmenter=None, **retrieve_multi_kw)` / `aretrieve_multi` / `aask_multi` | multi-query + LLM com integridade de citações |
| `ask(question, llm=..., citations=True, system_prompt=None, verify=None, verification_mode="report", segmenter=None, **retrieve_kw)` | → `Answer` (LLM é seu) |
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
`result.conflicts`, em `plan["eliminated"]` (com `explain=True`) e no trace, e
`ask_multi` os declara no prompt.

Três regras que só valem juntas:

- **Revogação é absoluta.** Documento cujo `status` está na classe revogada
  está fora de vigor por declaração própria — independentemente de o sucessor
  ter sido recuperado ou não. Julgada só de forma *relativa*, uma regra
  revogada que superasse o próprio sucessor entrava no contexto parecendo
  vigente, sem nada reportado. **Exceção:** se o filtro do chamador menciona o
  campo de status, ele está gerenciando status explicitamente e a regra
  absoluta não se aplica — é o que mantém "faceta histórica em `REVOGADO`"
  funcionando junto de `resolve_versions=True`.
- **Empate não é decisão.** Documentos que empatam no topo por (status, data,
  versão) são indistinguíveis *pelos metadados*: todos permanecem, e o
  conflito sai com `resolved: False` e `tied: [ids]`. Antes o desempate era
  alfabético por `document_id` — a evidência do perdedor sumia e a resposta
  parecia resolvida sem que nada a tivesse resolvido. `ask_multi` avisa o
  modelo: grupo não resolvido → relatar a divergência, não escolher um lado.
- **Eliminar não encolhe o contexto.** Cada chunk removido libera uma vaga que
  o próximo candidato ocupa, e a **garantia de cobertura é reservada de novo**
  sobre o que continua elegível. Sem isso, uma versão revogada que superasse a
  vigente custava as duas: a vaga da faceta era gasta na revogada, e a vigente
  ficava logo fora da janela que havia sido cortada para a perdedora. Fusão e
  resolução rodam até ponto fixo.

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
afirmação, `claim` + `citations` + `evidence` (documento, versão, **chunk_ids
reais**, metadados e o **texto do bloco citado**). Devolve um veredito por
afirmação — string ou dict com `verdict`, `rationale` e `quote` opcional.

**Texto da fonte junto da afirmação.** Cada `evidence` traz o `text` do bloco
`[n]` que a afirmação citou. Sem isso o juiz precisa reencontrar o bloco dentro
do `context` montado — e pode justificar a afirmação com um bloco que ela nunca
citou. O `context` completo continua no payload (julgar `uncited`, ou notar que
a evidência certa estava sob outro marcador, precisa dele); é o chamador quem
decide o que entra no prompt — veja os exemplos, que mostram as fontes citadas
por afirmação.

**Todo veredito nomeia seu lastro — e o lastro é conferido.** Um veredito é
uma afirmação *sobre* uma fonte, e cada um tem um lastro admissível:

| Veredito | Lastro |
|---|---|
| `supported` | as fontes citadas |
| `question_fact` | a pergunta |
| `inference`, `contradicted` | as fontes citadas **ou** a pergunta |
| `unsupported`, `uncited` | nenhum — afirmam uma *ausência* |

`quote` é comparado exatamente ao que o veredito invoca (substring, com
espaços e caixa normalizados — reformatar não muda de quem são as palavras):
fonte citada para `supported`, **a pergunta** para `question_fact`. Trecho que
não está lá é atribuição fabricada: a afirmação cai para `unsupported` e a
discrepância entra em `structural_issues`.

Sem `quote`, `require_evidence` (**ligado por padrão**) ainda exige que a
afirmação cite algo onde a fonte é lastro admissível. Sem isso, `supported`,
`inference` e `question_fact` eram **infalsificáveis**: rotular uma frase
inventada com qualquer um dos três passava com `ok=True`, sem citação, sem
trecho e sem nada conferido. `supported` sem citação vira `uncited` — a
afirmação pode até ser verdadeira, ela só não nomeou fonte nenhuma.

`ask()`/`ask_multi()` derivam `require_evidence` do próprio `citations`: quem
desligou as citações não é cobrado por elas. `require_quotes=True` vai além e
exige trecho para todo veredito com lastro; é opcional porque nem todo suporte
é um trecho contíguo (regra espalhada em duas frases, tabela), e exigi-lo
rejeitaria afirmações realmente sustentadas.

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
sustentada e a resposta ainda deixar uma faceta de fora. As facetas vão no
payload (`payload["facets"]`) e o verificador
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

**Facetas ≠ subconsultas.** `facets=[...]` declara o que a resposta **devia**
entregar, e vale tanto para o checklist no prompt quanto para o julgamento de
completude — a resposta é cobrada exatamente da lista que recebeu. Sem
`facets`, `ask_multi` usa as subconsultas, o que só é correto na medida em que
elas são atômicas (uma obrigação cada), que é a forma pedida ao decompositor.
Elas continuam sendo consultas de **recuperação**: um decompositor que divide
para *busca* (`"política de reembolso revisão 2024"`) cria obrigações que o
usuário nunca pediu, e cada uma vira uma faceta "não coberta". `facets=[]`
desliga o eixo (`complete=None`); `ask()` também aceita `facets=`, que é o que
torna `complete` utilizável fora do multi-query.

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

**Marcador depois do ponto final**: `... 30 dias. [1]` cita a mesma coisa que
`... 30 dias [1].` — o marcador é fonte da afirmação que ele **segue**. Antes o
corte acontecia no terminador, então o marcador caía na afirmação *seguinte*: a
afirmação que de fato citava a fonte voltava `uncited` (e `strict` apagava um
fato com fonte), a próxima recebia crédito de uma fonte que não citou, e um
`[2]` final virava uma "afirmação" que era só um marcador. Sem espaço
(`dias.[1] Já`) a resposta **não era dividida**. O marcador agora fica com sua
afirmação; um marcador no início da linha seguinte pertence àquela linha.

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
distinguir uma regra vigente de uma revogada. `Citation.text` traz o texto do
bloco, exatamente como ele aparece sob `[n]` no contexto.

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

### Verificador NLI offline (`ragvault.nli`)

Todo o resto da verificação é **estrutural**: a citação é substring da fonte, o
veredito nomeia um lastro, os spans são ordenados e não sobrepostos. A pergunta
que nenhuma delas responde é se a evidência citada de fato **acarreta** a
afirmação — e até aqui só um juiz LLM seu respondia, o que deixava a instalação
offline padrão sem nenhuma checagem de fidelidade.

```python
from ragvault.nli import nli_verifier

answer = kb.ask(question, llm=meu_llm, verify=nli_verifier())
```

Os três rótulos de NLI mapeiam nos vereditos existentes sem inventar nada:
`entailment → supported`, `contradiction → contradicted`, `neutral →
unsupported`. Instale com `pip install "ragvault[nli]"`.

| Parâmetro | Padrão | O que faz |
|---|---|---|
| `granularity` | `"sentence"` | divide o bloco citado em frases e pontua cada uma contra a afirmação (SummaC-ZS). `"block"` usa o chunk inteiro: mais barato, pior em premissa longa |
| `threshold` | `None` | `None` = `argmax` do próprio modelo. Um corte de probabilidade é prática consolidada, mas a constante não é transferível — derive a sua com `calibrate_threshold()` |
| `batch_size` / `max_length` | `16` / `512` | vazão em CPU |

**O que ele não faz**, dito abertamente porque um verificador que subnotifica
em silêncio é pior que nenhum:

- nunca devolve `question_fact` nem `inference` — ambos exigem ler a pergunta
  como fonte, o que NLI não faz. Afirmações que são inferência voltam
  `unsupported`;
- não reporta cobertura de facetas, então com facetas declaradas `complete`
  fica `False`. Passe `facets=[]`, ou combine com um juiz LLM para esse eixo;
- **a acurácia não é uniforme entre idiomas.** Um checkpoint multilíngue herda
  ~100 idiomas de pré-treino do XLM-R mas só os ~15 do fine-tuning em XNLI.
  Fora deles o comportamento é não medido.

**Agregação.** Uma frase-premissa que acarreta a afirmação já basta (`max`
existencial, não um botão de ajuste). Acarretamento é checado **antes** de
contradição: varrendo muitas frases, alguma não relacionada acaba pontuando
como contradição, e deixá-la vencer apagaria texto correto em `repair`. Os dois
erros não são simétricos — contradição perdida deixa uma frase errada visível,
contradição falsa remove em silêncio uma frase certa.

**Estado da medição:** ver `benchmarks/RESULTS-VERIFICATION.md`. Enquanto o
benchmark não tiver rodado com um modelo real, use em `report`/`annotate`, onde
um veredito errado é visível e não custa nada — **não** em `repair`/`strict`.

### Segmentação conectável (`segmenter=`)

O divisor embutido conhece terminadores de sentença — incluindo os que faltavam
(danda `।`, ponto etíope `።`, ponto armênio `։`, khmer `។`), sem os quais a
verificação por afirmação era um **no-op** em híndi, bengali, marathi, nepali,
amárico e armênio. Ele também tolera pontuação do mundo real: espaço faltando
depois do ponto (`"30 dias.Eles enviam"`), terminador ausente com quebra de
parágrafo, pontuação repetida — e deixou de quebrar em iniciais de nome
(`"John F. Kennedy"`).

O que nenhuma regra de terminador alcança é prosa que **não tem terminador**:
tailandês, laosiano, khmer e birmanês corridos. Para esses, passe o seu:

```python
import pysbd
seg = pysbd.Segmenter(language="en", clean=False)
answer = kb.ask(question, llm=meu_llm, verify=meu_verificador,
                segmenter=seg.segment)
```

Sem dependência nova no núcleo: como `verify=` e `rerank=`, é um callable seu.
As afirmações devolvidas passam pelo mesmo contrato estrutural do verificador —
substrings verbatim da resposta, em ordem, sem sobreposição — que é o que
mantém o `repair` cirúrgico. Um segmentador que quebra preserva a resposta e
registra o erro. `answer.verification.segmentation` diz qual rota foi usada:
`"heuristic"`, `"segmenter"` ou `"verifier"`.

**Não** dividimos em minúscula depois de ponto (`"30 dias. eles enviam"`): as
abreviaturas minúsculas são um conjunto aberto entre idiomas (`aprox.`, `pág.`,
`ca.`, `ex.`) e uma divisão falsa vira um fragmento que o `repair` apaga de um
texto que estava correto — pior que a divisão perdida.

### Quão específica precisa ser a `quote`

Estar **na** fonte não é o mesmo que apontar **para** algo nela: uma citação de
uma palavra ("o") é substring de quase qualquer documento.
`max_quote_occurrences` (8 por padrão) rejeita um span que casa tantas vezes no
bloco citado que não localiza nada. Conta ocorrências em vez de medir tamanho
de propósito: uma contagem significa a mesma coisa em qualquer script, enquanto
"pelo menos 4 caracteres" é uma oração em chinês e uma sílaba em finlandês.
Chunks duplicados não punem uma boa citação — as ocorrências são contadas por
fonte e minimizadas. `min_quote_coverage` (razão entre tamanho da citação e da
afirmação) é opcional e vem desligada.

**Trace** (`trace=True`): `subqueries`, `candidates_per_subquery`, `fusion`
(método, k0, contribuição de cada ranking por chunk, `fused_score` e
`coverage_reserved`), `version_conflicts`, `boosts`, `rerank`
(scores antes/depois, itens recuperados), `eliminated` (item + motivo),
`decompose_error`/`rerank_error` + fallback aplicado, e `stage_ms` por etapa.

## Exceções

`RagVaultError` (base Python), `ConfigurationError`, `EmbeddingError`, `IngestionError`, `EvaluationError`; nativas: `VaultError`, `VaultLockedError`, `VaultCorruptError`, `DimensionMismatchError`.
