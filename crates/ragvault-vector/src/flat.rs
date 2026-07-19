//! Flat (exact) index.
//!
//! Flat is a first-class backend: baseline, ground truth for recall tests,
//! the right answer for small collections and highly selective filters, and
//! the fallback when graph search cannot honor a filter. Filtering is a true
//! prefilter — rejected rows are never scored.

use rayon::prelude::*;

use crate::arena::VectorArena;
use crate::topk::TopK;

/// Minimum number of rows before parallel scan pays for thread overhead.
const PARALLEL_THRESHOLD: usize = 8192;

pub struct FlatIndex;

impl FlatIndex {
    /// Exact top-k over the arena. `accept` is the integrated filter
    /// predicate (tombstones are always excluded).
    pub fn search(
        arena: &VectorArena,
        prepared_query: &[f32],
        k: usize,
        accept: &(dyn Fn(u32) -> bool + Sync),
    ) -> Vec<(u32, f32)> {
        let n = arena.len();
        if n == 0 || k == 0 {
            return Vec::new();
        }
        if n < PARALLEL_THRESHOLD {
            let mut topk = TopK::new(k);
            for id in 0..n as u32 {
                if !arena.is_deleted(id) && accept(id) {
                    topk.push(id, arena.score(id, prepared_query));
                }
            }
            return topk.into_sorted();
        }
        // Parallel: per-shard bounded top-k, then merge — never a full sort.
        let shards = rayon::current_num_threads().max(1);
        let shard_size = n.div_ceil(shards);
        let merged = (0..shards)
            .into_par_iter()
            .map(|s| {
                let start = s * shard_size;
                let end = ((s + 1) * shard_size).min(n);
                let mut topk = TopK::new(k);
                for id in start as u32..end as u32 {
                    if !arena.is_deleted(id) && accept(id) {
                        topk.push(id, arena.score(id, prepared_query));
                    }
                }
                topk
            })
            .reduce(
                || TopK::new(k),
                |mut a, b| {
                    a.merge(b);
                    a
                },
            );
        merged.into_sorted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ragvault_core::Metric;

    fn build_arena(n: usize, dim: usize) -> VectorArena {
        let mut arena = VectorArena::new(dim, Metric::Cosine);
        let mut state = 12345u64;
        for _ in 0..n {
            let v: Vec<f32> = (0..dim)
                .map(|_| {
                    state ^= state << 13;
                    state ^= state >> 7;
                    state ^= state << 17;
                    ((state % 2000) as f32 / 1000.0) - 1.0
                })
                .collect();
            arena.push(&v).unwrap();
        }
        arena
    }

    #[test]
    fn exact_search_is_sorted_and_unique() {
        let arena = build_arena(500, 32);
        let q = arena.prepare_query(arena.get(7)).unwrap();
        let results = FlatIndex::search(&arena, &q, 10, &|_| true);
        assert_eq!(results.len(), 10);
        assert_eq!(results[0].0, 7, "self-query must rank itself first");
        for w in results.windows(2) {
            assert!(w[0].1 >= w[1].1, "results must be sorted best-first");
            assert_ne!(w[0].0, w[1].0);
        }
    }

    #[test]
    fn parallel_path_matches_serial() {
        let arena = build_arena(PARALLEL_THRESHOLD + 100, 16);
        let q = arena.prepare_query(arena.get(0)).unwrap();
        let parallel = FlatIndex::search(&arena, &q, 25, &|_| true);
        // serial reference over the same arena
        let mut topk = TopK::new(25);
        for id in 0..arena.len() as u32 {
            topk.push(id, arena.score(id, &q));
        }
        assert_eq!(parallel, topk.into_sorted());
    }

    #[test]
    fn filter_is_a_true_prefilter() {
        let arena = build_arena(200, 8);
        let q = arena.prepare_query(arena.get(3)).unwrap();
        let results = FlatIndex::search(&arena, &q, 50, &|id| id % 2 == 0);
        assert!(!results.is_empty());
        assert!(results.iter().all(|(id, _)| id % 2 == 0));
    }

    #[test]
    fn tombstones_are_invisible() {
        let mut arena = build_arena(50, 8);
        let q = arena.prepare_query(arena.get(5)).unwrap();
        arena.delete(5);
        let results = FlatIndex::search(&arena, &q, 50, &|_| true);
        assert!(results.iter().all(|(id, _)| *id != 5));
        assert_eq!(results.len(), 49);
    }
}
