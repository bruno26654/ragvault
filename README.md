# RagVault

**RagVault transforma documentos em uma base de conhecimento RAG de alta qualidade — busca híbrida, contexto inteligente, citações, avaliação e persistência durável em uma única API Python.**

De documentos a um RAG confiável em minutos, sem montar manualmente uma stack de várias bibliotecas.

```python
import ragvault

kb = ragvault.open(
    "./knowledge",
    preset="quality",
    embedding="sentence-transformers:all-MiniLM-L6-v2",  # decisão explícita
)
kb.sync("./documents")

result = kb.retrieve("Quais são as regras de cancelamento?", token_budget=6000)

print(result.context)    # contexto pronto para o modelo, com marcadores [n]
print(result.citations)  # citações com proveniência real (documento, versão, seção)
```

Opcionalmente, com um LLM plugado por você (RagVault nunca chama serviços externos sozinho):

```python
answer = kb.ask("Quais são as regras de cancelamento?", llm=my_llm, citations=True)
print(answer.text)
print(answer.citations)
```

## Por que RagVault

A maioria dos projetos RAG monta manualmente: parser + chunker + embeddings + vector store + BM25 + fusão + reranking + montagem de contexto + citações + avaliação. RagVault integra esse fluxo inteiro, com um núcleo Rust embutido (WAL, snapshots, HNSW, BM25, fusão RRF) atrás de uma API Python simples.

- **Python-first, local-first** — `pip install`, sem servidor, sem portas, sem Docker, sem downloads silenciosos de modelos.
- **Ingestão idempotente** — `kb.sync("./docs")` detecta novos/modificados/removidos por hash de conteúdo e atualiza atomicamente.
- **Busca híbrida** — denso (Flat/HNSW) + BM25 + sparse opcional, fundidos com RRF ponderado; filtros de metadados integrados à travessia do grafo (não pós-filtragem).
- **Contexto, não vizinhos** — agrupamento por documento, deduplicação, diversidade MMR, expansão de chunks vizinhos, token budget e citações estáveis.
- **Durabilidade real** — write-ahead log com checksum, snapshots publicados atomicamente, recuperação de crash testada.
- **Avaliação nativa** — `kb.evaluate("eval.jsonl")` com Recall@k, MRR, nDCG, latência p50/p95.

## Instalação

```bash
pip install ragvault
```

> **Estado atual (1.0.0-rc1):** release candidate CPU single-node. Wheels são construídas e testadas no CI para Linux x86-64/aarch64, macOS Apple Silicon e Windows x86-64, mas ainda não estão publicadas no PyPI; instale a partir do código-fonte com `maturin` (veja [Desenvolvimento](#desenvolvimento)) ou baixe os artefatos do CI. O pacote CPU não depende de CUDA.

Extras opcionais:

```bash
pip install "ragvault[pdf]"           # PDFs (pypdf)
pip install "ragvault[office]"        # DOCX (python-docx)
pip install "ragvault[local-models]"  # sentence-transformers
```

## Três níveis de uso

**Simples** — funciona offline com o embedder lexical embutido (determinístico, sem dependências; documentado como lexical, não semântico). O preset `quality` **exige** uma decisão explícita de embedding — nunca degrada silenciosamente para lexical nem baixa modelos sozinho:

```python
kb = ragvault.open("./data")  # preset balanced: lexical explícito por default
kb.add(["primeiro texto", "segundo texto"])
result = kb.retrieve("minha pergunta")
```

**Profissional** — presets, embeddings plugáveis, filtros e explain:

```python
kb = ragvault.open(
    "./data",
    preset="quality",
    embedding="sentence-transformers:all-MiniLM-L6-v2",
)
kb.sync("./documents", include=["**/*.md", "**/*.pdf"])
result = kb.retrieve(
    "minha pergunta",
    filters={"department": "legal", "year": {"gte": 2024}},
    token_budget=7000,
    explain=True,
)
print(result.plan)  # backend escolhido, razões, tempos
```

**Especialista** — todas as opções são explícitas, inspecionáveis e exportáveis:

```python
print(kb.config.explain())
kb.config.export("ragvault-config.json")
```

## Perguntas compostas (multi-query)

Uma pergunta com várias facetas perde recall numa consulta única: as facetas
competem pelo mesmo espaço de candidatos e a evidência de uma delas some.
`retrieve_multi`/`ask_multi` decompõem, buscam em lote, fundem com Weighted RRF
e **garantem cobertura por subconsulta**:

```python
answer = kb.ask_multi(
    "Qual o prazo para pedir reembolso e em quanto tempo o dinheiro volta?",
    llm=answer_llm,              # seu provedor (callable)
    decompose=query_decomposer,  # opcional; falha → cai para consulta única
    filters={"status": "VIGENTE"},
    resolve_versions=True,       # revogados nunca são citados
    trace=True,
)

print(answer.text)                     # [n] inexistentes são removidos
print(answer.result.subqueries)        # o que foi realmente buscado
print(answer.result.conflicts)         # versões superadas, explicitamente
```

Medido em `benchmarks/RESULTS-MULTIQUERY.md` (24 perguntas multi-hop):
full-recall **0,167 → 0,875**, recall 0,493 → 0,951, a 1,02× dos tokens
enviados ao LLM. Exemplo completo (com Groq opcional):
[`examples/multi_query_rag.py`](examples/multi_query_rag.py).

## Embeddings plugáveis

```python
kb = ragvault.open("./data", embedding=my_embedding_function)          # callable
kb = ragvault.open("./data", embedding="sentence-transformers:nome")   # extra opcional
kb = ragvault.open("./data", embedding="builtin:hashed-ngram:512")     # offline default
```

Embeddings são cacheados por `(hash do conteúdo, modelo, configuração)` em SQLite — re-sincronizações não reprocessam o que não mudou.

## CLI

```bash
ragvault init ./data --preset quality
ragvault sync ./data ./documents
ragvault query ./data "minha pergunta" --explain
ragvault doctor ./data
ragvault evaluate ./data evaluation.jsonl
ragvault compact ./data
ragvault studio ./data        # UI local de inspeção (sem dependências extras)
```

## Multi-tenancy

```python
tenant_kb = kb.for_tenant("acme")
tenant_kb.add(...)        # gravações etiquetadas automaticamente
tenant_kb.retrieve(...)   # filtro de tenant aplicado a toda query
```

## Status dos recursos

| Recurso | Status |
|---|---|
| KnowledgeBase (`open/add/sync/retrieve/ask/evaluate`) | implemented |
| Atomicidade de escrita (prepared-write pré-WAL, replay de batch corrompido falha claro) | implemented + testes de regressão |
| Equivalência compact == compact+reopen (suíte diferencial, todos os backends) | implemented |
| Identidade de ingestão por sha256(bytes) + fingerprints de pipeline | implemented |
| Preset `offline-lite` (baseline lexical explícito); `quality` exige decisão de embedding | implemented |
| Contexto v2: fusão de chunks adjacentes + flag `result.truncated` | implemented |
| Flat exato + HNSW com filtros integrados | implemented |
| BM25 incremental + fusão RRF ponderada | implemented |
| Sparse vectors fornecidos pelo usuário | implemented (persistidos no WAL, sobrevivem a crash e compactação) |
| WAL + recovery + snapshots atômicos | implemented |
| Contexto com MMR, expansão, token budget, citações | implemented |
| Avaliação nativa (Recall@k, MRR, nDCG, latência) | implemented |
| CLI (`init/sync/query/doctor/evaluate/compact`) | implemented |
| Presets (`quality/balanced/fast/offline/...`) | implemented |
| Comparação de presets e auto-tuning (`kb.compare`/`kb.tune`/`kb.apply`) | implemented |
| Integração LangChain (`kb.as_langchain_retriever()`) | implemented |
| Quantização SQ8 (int8 scan + rescore f32, preset `low_memory`) | implemented |
| `kb.migrate_embeddings` (estratégia blocking, swap atômico) | implemented |
| RagVault Studio (`ragvault studio` — UI local sem dependências) | implemented |
| IVF-Flat / IVF-PQ (`index="ivf_flat"\|"ivf_pq"`) | implemented |
| mmap (`storage="mmap"`) | implemented |
| Interop Faiss (`ragvault.compat.faiss`, nível convertible) | implemented |
| MaxSim reranking (`ragvault.maxsim_reranker`) | implemented |
| `Database` multi-coleção | implemented |
| GPU CAGRA sidecar (`ragvault.gpu`) | experimental — implementado com testes de plumbing; **não validado em GPU real** (runbook: docs/GPU.md) |
| Integrações LlamaIndex/Haystack/DSPy | implemented — roundtrips reais testados no CI com versões fixadas (job `integrations`) |
| Storage v2: base binária segmentada + flush O(delta) + compactação read-friendly (ADR 0016) | implemented — CRC por registro/stream, migração v1→v2 transparente, leitores não bloqueiam durante a compactação |
| Filtros tipados de metadados (posting lists + ranges) | implemented — até 484× em seletividade 0.1% (benchmarks/RESULTS-FILTERS.md) |
| Batch nativo (`kb.retrieve_many`, GIL liberado, paralelo por query) | implemented |
| Multi-query (`kb.retrieve_multi` / `kb.ask_multi`): decomposição, Weighted RRF global, cobertura por subconsulta, precedência de versões, citações verificadas | implemented — full-recall 0.167 → 0.875 em perguntas compostas (benchmarks/RESULTS-MULTIQUERY.md) |
| Wheels multiplataforma no PyPI | planned (CI configurado para Linux) |

Detalhes de arquitetura interna (segmentos, HNSW, WAL): [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Limitações conhecidas: [docs/TROUBLESHOOTING.md](docs/TROUBLESHOOTING.md).

## Desenvolvimento

```bash
git clone https://github.com/bruno26654/ragvault.git && cd ragvault
python -m venv .venv && source .venv/bin/activate
pip install maturin pytest numpy
maturin develop --release      # compila o núcleo Rust e instala o pacote
cargo test --workspace --exclude ragvault-python   # testes Rust
python -m pytest               # testes Python end-to-end
python benchmarks/bench_retrieval.py               # benchmarks reais
```

## Licença

Apache-2.0
