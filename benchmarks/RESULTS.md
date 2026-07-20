# RagVault benchmark results

- date: 2026-07-20 00:50:17 UTC
- host: Linux-6.18.5-x86_64-with-glibc2.39, x86_64, python 3.11.15
- cpus: 4

## Engine ANN benchmark (n=50,000, dim=384, k=10, 100 queries)

> Dataset: random gaussian vectors (normalized). This is an adversarial case for ANN recall — distances concentrate in high dimensions — so absolute recall is low for every HNSW implementation; the comparison between systems at equal ef is the meaningful signal. The RAG benchmark below uses text.

- ingestion (WAL batch + incremental HNSW build): 128.3s (390 vectors/s)

| backend | recall@10 | QPS (1 thread) | p50 ms | p95 ms |
|---|---|---|---|---|
| RagVault HNSW ef=64 | 0.206 | 777 | 1.23 | 1.96 |
| RagVault HNSW ef=128 | 0.312 | 541 | 1.82 | 2.19 |
| RagVault HNSW ef=256 | 0.450 | 315 | 3.12 | 3.61 |

- flush (snapshot publish): 1.37s; reopen from snapshot: 0.20s; post-reopen recall@10 (ef=128): 0.312
- on-disk size: 100 MB; peak RSS: 379 MB

### Comparison with faiss-cpu 1.14.3 (same dataset, same queries, same k, in-process, 1 query thread)

| backend | recall@10 | QPS (1 thread) | p50 ms | p95 ms | build s |
|---|---|---|---|---|---|
| faiss HNSW ef=64 | 0.214 | 237 | 0.40 | 0.63 | 40.9 |
| faiss HNSW ef=128 | 0.345 | 1,662 | 0.60 | 0.70 | 40.9 |
| faiss HNSW ef=256 | 0.486 | 939 | 1.05 | 1.16 | 40.9 |

> Fairness notes: identical vectors, queries, k and machine. faiss timing covers only `index.search`; RagVault timing crosses the Python/JSON binding and includes filter/tombstone checks — this overhead is included on purpose because it is what a user observes. RagVault ingestion includes WAL durability; faiss build does not persist anything.

## SQ8 quantized backend (n=50,000, dim=384, k=10)

- ingestion (WAL + quantize, NO graph build): 0.4s (129,501 vectors/s)
- search (int8 scan 4x oversample + f32 rescore): recall@10 1.000, QPS 99, p50 9.99 ms, p95 10.36 ms
- quantized scan memory: 19 MB vs 77 MB f32 (4.0x smaller); f32 kept for rescoring
- trade-off vs HNSW at this scale: near-exact recall and ~300x faster durable ingestion (no graph build), at the cost of an O(n) scan per query (slower QPS than HNSW here) — the right choice for write-heavy or filter-heavy medium collections

## End-to-end RAG benchmark (2,000 text documents, hybrid retrieval)

- ingestion (parse + chunk + embed + WAL + index): 13.1s (153 docs/s, embedder = builtin hashed-ngram)
- hybrid retrieve() incl. context assembly: p50 4.6 ms, p95 4.8 ms, p99 4.9 ms
- filtered retrieve() (10% selectivity): p50 5.8 ms, p95 6.0 ms
- reopen (snapshot load): 0.01s; retrieval works after reopen
