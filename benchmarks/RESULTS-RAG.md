# RAG quality evaluation — real text

- date: 2026-07-20 17:18 UTC
- host: Linux-6.18.5-x86_64-with-glibc2.39, python 3.11.15
- dataset: 30 passages / 24 queries (12 paraphrase + 12 keyword), committed in benchmarks/data/
- k = 5; every row below was actually executed on this machine

| config | recall@k | MRR | nDCG@k | precision@k | MRR paraphrase | MRR keyword | dup rate | ctx tokens | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 only | 0.792 | 0.706 | 0.727 | 0.352 | 0.413 | 1.000 | 0.000 | 123 | 0.6 | 2.7 |
| lexical dense only | 0.833 | 0.708 | 0.740 | 0.167 | 0.417 | 1.000 | 0.000 | 169 | 2.3 | 2.5 |
| hybrid (lexical+bm25) | 0.708 | 0.654 | 0.667 | 0.142 | 0.308 | 1.000 | 0.000 | 167 | 2.3 | 2.7 |
| hybrid + expansion | 0.708 | 0.654 | 0.667 | 0.142 | 0.308 | 1.000 | 0.000 | 167 | 2.5 | 2.7 |
| hybrid + MMR(0.5) | 0.708 | 0.654 | 0.667 | 0.142 | 0.308 | 1.000 | 0.000 | 167 | 2.4 | 3.9 |

> Semantic rows (semantic dense / semantic hybrid / + rerank) were
> NOT run in this environment: the network policy blocks
> huggingface.co (CONNECT 403), so the model cannot be downloaded.
> Run them elsewhere with:
> `pip install "ragvault[local-models]" && python benchmarks/bench_rag_quality.py --semantic`

## Reading the numbers honestly

- Measured observation: hybrid RRF slightly *underperforms* the best single
  signal here because both signals (hashed-ngram and BM25) are lexical and
  highly correlated — fusing them dilutes rank-1 hits on paraphrase queries.
  Hybrid earns its keep when the signals are complementary (semantic dense +
  BM25), which is exactly the semantic rows blocked in this environment.
- The paraphrase-MRR column is the semantic gap: hashed-ngram is a
  lexical projection, so paraphrased queries (low word overlap) are
  where it underperforms and where a semantic model earns its cost.
- Keyword-MRR shows BM25/hybrid doing what lexical retrieval is good
  at. The `quality` preset therefore refuses to run without an
  explicit embedding decision.
