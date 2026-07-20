# RagVault benchmark results

- date: 2026-07-20 07:40:04 UTC
- host: Linux-6.18.5-x86_64-with-glibc2.39, x86_64, python 3.11.15
- cpus: 4

## Engine ANN benchmark (n=50,000, dim=384, k=10, 100 queries)

> Dataset: random gaussian vectors (normalized). This is an adversarial case for ANN recall — distances concentrate in high dimensions — so absolute recall is low for every HNSW implementation; the comparison between systems at equal ef is the meaningful signal. The RAG benchmark below uses text.

- ingestion (WAL batch + incremental HNSW build): 158.3s (316 vectors/s)

| backend | recall@10 | QPS (1 thread) | p50 ms | p95 ms |
|---|---|---|---|---|
| RagVault HNSW ef=64 | 0.206 | 802 | 1.22 | 1.47 |
| RagVault HNSW ef=128 | 0.312 | 441 | 2.24 | 2.55 |
| RagVault HNSW ef=256 | 0.450 | 261 | 3.79 | 4.19 |

- flush (snapshot publish): 1.80s; reopen from snapshot: 0.22s; post-reopen recall@10 (ef=128): 0.312
- on-disk size: 100 MB; peak RSS: 380 MB

### Comparison with faiss-cpu 1.14.3 (same dataset, same queries, same k, in-process, 1 query thread)

| backend | recall@10 | QPS (1 thread) | p50 ms | p95 ms | build s |
|---|---|---|---|---|---|
| faiss HNSW ef=64 | 0.214 | 938 | 0.40 | 0.51 | 43.6 |
| faiss HNSW ef=128 | 0.345 | 1,433 | 0.68 | 0.82 | 43.6 |
| faiss HNSW ef=256 | 0.486 | 792 | 1.26 | 1.38 | 43.6 |

> Fairness notes: identical vectors, queries, k and machine. faiss timing covers only `index.search`; RagVault timing crosses the Python/JSON binding and includes filter/tombstone checks — this overhead is included on purpose because it is what a user observes. RagVault ingestion includes WAL durability; faiss build does not persist anything.

## SQ8 quantized backend (n=50,000, dim=384, k=10)

- ingestion (WAL + quantize, NO graph build): 0.4s (134,162 vectors/s)
- search (int8 scan 4x oversample + f32 rescore): recall@10 1.000, QPS 76, p50 13.08 ms, p95 13.31 ms
- quantized scan memory: 19 MB vs 77 MB f32 (4.0x smaller); f32 kept for rescoring
- trade-off vs HNSW at this scale: near-exact recall and ~300x faster durable ingestion (no graph build), at the cost of an O(n) scan per query (slower QPS than HNSW here) — the right choice for write-heavy or filter-heavy medium collections

## IVF-Flat backend (n=50,000, dim=384, k=10)

- ingestion (WAL only, no per-insert index work): 0.3s (169,630 vectors/s); flush incl. k-means train + assign: 12.6s

| nprobe | recall@10 | QPS (1 thread) | p50 ms | p95 ms |
|---|---|---|---|---|
| 4 | 0.090 | 1,985 | 0.49 | 0.66 |
| 8 | 0.144 | 1,193 | 0.84 | 0.92 |
| 16 | 0.222 | 651 | 1.53 | 1.65 |
| 32 | 0.376 | 337 | 2.89 | 3.67 |
| 64 | 0.545 | 169 | 5.80 | 6.79 |

> Same adversarial gaussian dataset as above: absolute recall is low for every ANN method here; the signal is recall growing monotonically with nprobe and the latency/recall trade-off. `nprobe` is per-query (`kb.retrieve(..., nprobe=...)`).

## End-to-end RAG benchmark (2,000 text documents, hybrid retrieval)

- ingestion (parse + chunk + embed + WAL + index): 13.0s (154 docs/s, embedder = builtin hashed-ngram)
- hybrid retrieve() incl. context assembly: p50 5.6 ms, p95 6.3 ms, p99 6.8 ms
- filtered retrieve() (10% selectivity): p50 6.4 ms, p95 7.2 ms
- reopen (snapshot load): 0.02s; retrieval works after reopen
