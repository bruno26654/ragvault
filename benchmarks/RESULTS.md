# RagVault benchmark results

- date: 2026-07-19 19:47:49 UTC
- host: Linux-6.18.5-x86_64-with-glibc2.39, x86_64, python 3.11.15
- cpus: 4

## Engine ANN benchmark (n=50,000, dim=384, k=10, 100 queries)

> Dataset: random gaussian vectors (normalized). This is an adversarial case for ANN recall — distances concentrate in high dimensions — so absolute recall is low for every HNSW implementation; the comparison between systems at equal ef is the meaningful signal. The RAG benchmark below uses text.

- ingestion (WAL batch + incremental HNSW build): 126.9s (394 vectors/s)

| backend | recall@10 | QPS (1 thread) | p50 ms | p95 ms |
|---|---|---|---|---|
| RagVault HNSW ef=64 | 0.206 | 1,152 | 0.86 | 1.05 |
| RagVault HNSW ef=128 | 0.312 | 660 | 1.50 | 1.70 |
| RagVault HNSW ef=256 | 0.450 | 380 | 2.62 | 2.93 |

- flush (snapshot publish): 0.93s; reopen from snapshot: 0.22s; post-reopen recall@10 (ef=128): 0.312
- on-disk size: 100 MB; peak RSS: 380 MB

### Comparison with faiss-cpu 1.14.3 (same dataset, same queries, same k, in-process, 1 query thread)

| backend | recall@10 | QPS (1 thread) | p50 ms | p95 ms | build s |
|---|---|---|---|---|---|
| faiss HNSW ef=64 | 0.214 | 3,721 | 0.26 | 0.33 | 34.4 |
| faiss HNSW ef=128 | 0.345 | 2,108 | 0.47 | 0.55 | 34.4 |
| faiss HNSW ef=256 | 0.486 | 1,187 | 0.84 | 0.93 | 34.4 |

> Fairness notes: identical vectors, queries, k and machine. faiss timing covers only `index.search`; RagVault timing crosses the Python/JSON binding and includes filter/tombstone checks — this overhead is included on purpose because it is what a user observes. RagVault ingestion includes WAL durability; faiss build does not persist anything.

## End-to-end RAG benchmark (2,000 text documents, hybrid retrieval)

- ingestion (parse + chunk + embed + WAL + index): 14.5s (138 docs/s, embedder = builtin hashed-ngram)
- hybrid retrieve() incl. context assembly: p50 5.5 ms, p95 6.0 ms, p99 6.6 ms
- filtered retrieve() (10% selectivity): p50 6.4 ms, p95 8.0 ms
- reopen (snapshot load): 0.02s; retrieval works after reopen
