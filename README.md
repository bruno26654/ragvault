# RagVault

**RagVault transforma documentos em uma base de conhecimento RAG de alta qualidade — busca híbrida, contexto inteligente, citações, avaliação e persistência durável em uma única API Python.**

De documentos a um RAG confiável em minutos, sem montar manualmente uma stack de várias bibliotecas.

```python
import ragvault

kb = ragvault.open("./knowledge", preset="quality")
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

> **Estado atual (0.1):** wheels pré-compiladas ainda não estão publicadas no PyPI; instale a partir do código-fonte com `maturin` (veja [Desenvolvimento](#desenvolvimento)). O pacote CPU não depende de CUDA.

Extras opcionais:

```bash
pip install "ragvault[pdf]"           # PDFs (pypdf)
pip install "ragvault[office]"        # DOCX (python-docx)
pip install "ragvault[local-models]"  # sentence-transformers
```

## Três níveis de uso

**Simples** — funciona offline com o embedder lexical embutido (determinístico, sem dependências; documentado como lexical, não semântico — combine com um modelo real para máxima qualidade):

```python
kb = ragvault.open("./data")
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
| Integrações LlamaIndex/Haystack/DSPy | implemented (untested: requerem as libs; erro acionável sem elas) |
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
