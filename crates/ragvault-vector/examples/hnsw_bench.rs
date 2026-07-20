//! HNSW search microbenchmark for the `visited`-set profiling work.
//!
//! Run: `cargo run --release --example hnsw_bench -p ragvault-vector`
//!
//! Builds a graph over N vectors, times Q queries at a fixed ef, and reports
//! mean/p50/p95 latency, QPS and recall@k against exact Flat ground truth.
//! Used to prove (or disprove) that a change to the search hot path is a real
//! speedup with no recall regression — not to publish universal numbers.

use std::time::Instant;

use ragvault_core::Metric;
use ragvault_vector::flat::FlatIndex;
use ragvault_vector::hnsw::{build_from_arena, random_vectors};
use ragvault_vector::{HnswConfig, VectorArena};

fn main() {
    let n: usize = std::env::var("BENCH_N")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50_000);
    let dim: usize = std::env::var("BENCH_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64);
    let queries: usize = std::env::var("BENCH_Q")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);
    let ef: usize = std::env::var("BENCH_EF")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);
    let k: usize = 10;

    eprintln!("building arena: n={n} dim={dim}");
    let mut arena = VectorArena::new(dim, Metric::Cosine);
    for v in random_vectors(n, dim, 7) {
        arena.push(&v).unwrap();
    }

    let build_start = Instant::now();
    let hnsw = build_from_arena(&arena, HnswConfig::default());
    let build_secs = build_start.elapsed().as_secs_f64();
    eprintln!("built graph in {build_secs:.2}s");

    let qs = random_vectors(queries, dim, 999);
    let prepared: Vec<Vec<f32>> = qs.iter().map(|q| arena.prepare_query(q).unwrap()).collect();

    // Warm up (touch pages, prime any thread-local scratch).
    for p in prepared.iter().take(50) {
        let _ = hnsw.search(&arena, p, k, ef, None);
    }

    // Recall vs exact Flat.
    let mut hits = 0usize;
    let mut total = 0usize;
    for p in &prepared {
        let truth: Vec<u32> = FlatIndex::search(&arena, p, k, &|_| true)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        let approx: Vec<u32> = hnsw
            .search(&arena, p, k, ef, None)
            .into_iter()
            .map(|(id, _)| id)
            .collect();
        for id in &approx {
            if truth.contains(id) {
                hits += 1;
            }
        }
        total += truth.len();
    }
    let recall = hits as f64 / total as f64;

    // Latency.
    let mut lat_us: Vec<f64> = Vec::with_capacity(prepared.len());
    let wall = Instant::now();
    for p in &prepared {
        let t = Instant::now();
        let _ = hnsw.search(&arena, p, k, ef, None);
        lat_us.push(t.elapsed().as_secs_f64() * 1e6);
    }
    let wall_secs = wall.elapsed().as_secs_f64();
    lat_us.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = lat_us.iter().sum::<f64>() / lat_us.len() as f64;
    let p50 = lat_us[lat_us.len() / 2];
    let p95 = lat_us[(lat_us.len() as f64 * 0.95) as usize];
    let p99 = lat_us[(lat_us.len() as f64 * 0.99) as usize];
    let qps = prepared.len() as f64 / wall_secs;

    println!("n={n} dim={dim} ef={ef} k={k} queries={queries}");
    println!("recall@{k}={recall:.4}");
    println!("latency_us mean={mean:.1} p50={p50:.1} p95={p95:.1} p99={p99:.1}");
    println!("qps={qps:.0}");
}
