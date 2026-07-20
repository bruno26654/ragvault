# Typed-filter selectivity benchmark

- 2026-07-20 17:23 UTC, Linux-6.18.5-x86_64-with-glibc2.39, n=20,000, dim=64, k=10, 60 queries, dense over HNSW-eligible collection

| selectivity | typed prefilter p50/p95 ms | backend | predicate-path p50/p95 ms | backend | speedup (p50) |
|---|---|---|---|---|---|
| 100% | 0.25 / 0.33 | hnsw | 0.35 / 0.43 | hnsw | 1.4x |
| 50% | 0.43 / 0.67 | hnsw | 0.88 / 1.06 | hnsw | 2.1x |
| 10% | 0.19 / 0.32 | bitmap_prefiltered_flat | 2.22 / 2.68 | hnsw | 11.9x |
| 1% | 0.05 / 0.07 | bitmap_prefiltered_flat | 8.36 / 12.21 | hnsw | 172.1x |
| 0.1% | 0.04 / 0.06 | bitmap_prefiltered_flat | 17.60 / 20.41 | hnsw | 483.8x |

> Both columns run semantically identical filters; the predicate column uses a non-extractable shape ($or of the same clause) to force per-candidate JSON evaluation. Typed path switches to bitmap_prefiltered_flat when the candidate set is small; selectivity and coverage are reported in every plan.
