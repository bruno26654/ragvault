# RAG quality evaluation — real text

- date: 2026-08-02 14:45 UTC
- host: Linux-6.14.0-37-generic-x86_64-with-glibc2.39, python 3.12.3
- dataset: 30 passages / 24 queries (12 paraphrase + 12 keyword), committed in benchmarks/data/
- k = 5; every row below was actually executed
- semantic rows: `sentence-transformers` model, run on a machine with model
  access (`pip install "ragvault[local-models]" && python benchmarks/bench_rag_quality.py --semantic`)

| config | recall@k | MRR | nDCG@k | precision@k | MRR paraphrase | MRR keyword | dup rate | ctx tokens | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|---|---|---|
| bm25 only | 0.792 | 0.706 | 0.727 | 0.352 | 0.413 | 1.000 | 0.000 | 123 | 0.6 | 6.6 |
| lexical dense only | 0.833 | 0.708 | 0.740 | 0.167 | 0.417 | 1.000 | 0.000 | 169 | 2.2 | 2.4 |
| hybrid (lexical+bm25) | 0.708 | 0.654 | 0.667 | 0.142 | 0.308 | 1.000 | 0.000 | 167 | 2.3 | 2.4 |
| hybrid + expansion | 0.708 | 0.654 | 0.667 | 0.142 | 0.308 | 1.000 | 0.000 | 167 | 2.4 | 4.0 |
| hybrid + MMR(0.5) | 0.708 | 0.654 | 0.667 | 0.142 | 0.308 | 1.000 | 0.000 | 167 | 2.2 | 6.0 |
| **semantic dense** | **1.000** | **0.979** | **0.985** | 0.200 | **0.958** | 1.000 | 0.000 | 167 | 10.1 | 11.7 |
| semantic hybrid | 0.833 | 0.785 | 0.797 | 0.167 | 0.569 | 1.000 | 0.000 | 164 | 10.3 | 12.4 |
| semantic hybrid + rerank | 1.000 | 0.979 | 0.985 | 0.200 | 0.958 | 1.000 | 0.000 | 167 | 411.9 | 472.4 |

## Reading the numbers honestly

**The semantic gap is the headline.** Paraphrase-MRR goes from 0.417 (lexical
dense) to **0.958** (semantic dense) — an embedding model more than doubles
retrieval quality on questions phrased differently from the source text, and
takes recall@5 to 1.000. This is exactly why the `quality` preset refuses to
run without an explicit embedding decision instead of silently falling back to
the lexical baseline: on keyword queries the two are indistinguishable
(MRR 1.000 either way), so the gap only shows up where users actually notice it.

**Hybrid RRF hurts here — twice, for the same reason.** Fusing lexical dense
with BM25 drops MRR from 0.708 to 0.654, and fusing *semantic* dense with BM25
drops it much harder, from 0.979 to 0.785. RRF assumes the signals are
complementary; when one signal is decisively better, fusion drags it toward the
weaker one. Take this as a warning against enabling hybrid reflexively: measure
on your own corpus, and prefer pure dense when your embedding model is strong.

**Reranking cost 40x latency for zero quality gain.** `semantic hybrid + rerank`
is identical to `semantic dense` on every quality metric (recall 1.000, MRR
0.979, nDCG 0.985) while p50 goes from 10.1 ms to **411.9 ms** and p95 to
472.4 ms. On this corpus the reranker only repairs the damage hybrid fusion
caused — it buys back exactly what fusion lost, at 40x the cost. A reranker is
worth its price when it fixes ordering the retriever genuinely got wrong; here
the retriever was already right. Always measure before shipping one.

**Where lexical still wins:** BM25 has the best precision@5 (0.352) and the
smallest context (123 tokens) because it returns fewer, tighter matches, and it
is ~17x faster than the semantic path (0.6 ms vs 10.1 ms p50). For
keyword-shaped queries it ties the semantic model at MRR 1.000. `offline-lite`
is a real option when queries are keyword-like and no download is acceptable.

Reproduce: `python benchmarks/bench_rag_quality.py` (offline rows) and
`pip install "ragvault[local-models]" && python benchmarks/bench_rag_quality.py --semantic`
(all rows).
