# Retrieval

## Modos

`mode="auto"` (default via preset): híbrido quando há texto e vetor; `dense`, `keyword`, `sparse` disponíveis explicitamente.

## Sinais

- **Dense**: Flat exato abaixo de `flat_threshold` (default 1000 vetores) ou HNSW (M=16, ef_construction=200). `ef_search` configurável por query.
- **BM25**: k1=1.2, b=0.75, tokenizer Unicode lowercase, estatísticas sobre docs vivos.
- **Sparse**: vetores esparsos fornecidos por você (SPLADE, BGE-M3...); o core nunca gera sparse embeddings.

### IVF (`index="ivf_flat"` / `index="ivf_pq"`)

Particionamento k-means (nlist automático = √n, limitado a [16, 1024]) com
`nprobe` listas sondadas por query. O índice é uma estrutura de aceleração
reconstruível: treinado em open/flush/compact; linhas inseridas depois do
treino são varridas exatamente (delta scan) até o próximo rebuild — escritas
novas nunca ficam invisíveis. Com PQ (`ivf_pq`), códigos de 8 bits por
subespaço + tabelas ADC + oversampling 8× + rescore f32. Abaixo de 256 linhas
o planner usa Flat exato e diz isso no plan.

## Fusão

RRF ponderado: `score = Σ w_s / (60 + rank_s)`. Rank-based — escalas de BM25 e cosseno nunca são somadas diretamente. Pesos em `dense_weight`/`bm25_weight`/`sparse_weight`.

## Filtros

DSL: `eq` implícito, `ne, in, not_in, gt, gte, lt, lte, contains, contains_any, contains_all, exists, prefix`, composição `$and/$or/$not`, paths com ponto (`"a.b"`). Semântica de ausente/null/NaN documentada em ADR 0010. Operador desconhecido = erro imediato.

Filtros são integrados: no Flat são prefilter verdadeiro; no HNSW a travessia usa nós reprovados como pontes sem retorná-los, amplia `ef` quando faminta e cai para Flat exato filtrado se necessário — tudo visível em `result.plan.reason`.

## Explain e trace

```python
result = kb.retrieve("pergunta", explain=True, trace=True)
result.plan   # backend, razões, ef, pool, tempos
result.trace  # contadores da montagem de contexto, erros de rerank, ms de busca
```

## Reranking

```python
result = kb.retrieve(query, rerank=my_reranker)  # callable(query, chunks) -> chunks
```

Falha do reranker mantém o ranking anterior (modo tolerante) e registra o motivo no trace.
