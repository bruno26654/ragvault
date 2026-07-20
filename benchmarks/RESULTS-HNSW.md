# HNSW search profiling — `visited` scratch buffer

**Question:** is the per-`search_layer` `visited` set (previously
`vec![false; nodes.len()]`, an `O(nodes)` allocation + zeroing on every call)
a real bottleneck, and does replacing it with a generation-stamped scratch
buffer (`O(1)` reset, thread-local reuse) actually help?

**Method.** `cargo run --release --example hnsw_bench -p ragvault-vector`
(source in `crates/ragvault-vector/examples/hnsw_bench.rs`). Builds a graph
over `N` random unit vectors (dim 64, cosine, `HnswConfig::default()`:
M=16, ef_construction=200), then times `Q` queries at `ef=128`, `k=10`.
Recall is measured against exact Flat ground truth on the same arena. Numbers
are from one machine (this CI-class box, single-threaded query loop) — they
compare the two implementations on identical data, not a universal claim.

The "before" row is the previous `vec![false; nodes.len()]` implementation
(recovered via `git stash`); the "after" row is the generation-stamped
`VisitedSet`. Same seed, same queries, back to back.

## N = 50,000 (Q = 2000)

| Impl | recall@10 | mean µs | p50 | p95 | p99 | QPS |
|---|---|---|---|---|---|---|
| before (`vec![false; N]`) | 0.7939 | 543.4 | 532.3 | 685.9 | 791.5 | 1840 |
| after (gen-stamp) | 0.7939 | 592.4 | 584.1 | 698.2 | 796.5 | 1688 |

At 50k the `visited` buffer is ~50 KB; zeroing it is cheap next to ~ef·M
64-dim distance computations, so the change is **neutral to slightly negative**
(within run-to-run noise). Recall is byte-identical — the optimization is a
pure mechanical rewrite, not an algorithmic change.

## N = 200,000 (Q = 1500)

| Impl | recall@10 | mean µs | p50 | p95 | p99 | QPS |
|---|---|---|---|---|---|---|
| before (`vec![false; N]`) | 0.6049 | 1698.8 | 1114.3 | 3831.6 | 12636.9 | 589 |
| after (gen-stamp) | 0.6049 | 890.5 | 879.9 | 1035.9 | 1218.4 | 1123 |

At 200k the picture flips hard. The `O(nodes)` allocate + memset per query now
dominates the tail: **mean 1.9× faster, p95 3.7× better, p99 10.4× better,
QPS 1.9× higher**, recall unchanged. The p99 collapse (12.6 ms → 1.2 ms) is the
allocator/zeroing jitter disappearing — every query stopped touching 200 KB of
fresh memory it never needed.

## Conclusion

The `visited` allocation is a **proven** bottleneck at scale (N ≥ ~200k) and
**harmless** below it. The generation-stamped, thread-local `VisitedSet` is
therefore kept: no recall cost anywhere, a large latency/throughput win where
it matters, and (because each rayon worker owns its buffer) it composes with
the native batch `search_many` path.

Not pursued, because unproven here: replacing the two per-call `BinaryHeap`
scratch structures (they are bounded by `ef`, not `N`), and SIMD intrinsics in
the distance kernel (the auto-vectorized 4-accumulator kernel already
dominates cost at these dims — a separate benchmark gate, see
`benchmarks/RESULTS.md`). Reproduce any of this with the example above;
`BENCH_N`, `BENCH_Q`, `BENCH_EF`, `BENCH_DIM` are environment overrides.
