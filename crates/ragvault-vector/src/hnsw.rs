//! HNSW (Hierarchical Navigable Small World) graph index.
//!
//! Design notes:
//! - Distances come from the shared [`VectorArena`]; the graph stores only
//!   compact adjacency lists (`Vec<u32>` per node per layer).
//! - Insertion uses the classic algorithm: greedy descent through upper
//!   layers, `ef_construction` beam search per level, neighbor selection with
//!   the diversity heuristic, bidirectional linking with degree-bounded
//!   pruning.
//! - Deletions tombstone the row in the arena. Tombstoned nodes still serve
//!   as graph bridges during traversal but are never returned.
//! - Filtered search is traversal-aware: the beam only counts nodes that
//!   pass the predicate toward the result set, and when the filter is
//!   restrictive the caller retries with a larger `ef` or falls back to a
//!   filtered Flat scan (decided by the engine's planner).
//! - Invariants tested below: no self-loops, no duplicate neighbors, degree
//!   bounds, valid entry point, recall vs Flat, serialization round-trip.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

use crate::arena::VectorArena;
use crate::topk::TopK;

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct HnswConfig {
    /// Max connections per node on layers > 0. Layer 0 allows `2 * m`.
    pub m: usize,
    pub ef_construction: usize,
    /// Default `ef` at query time (callers can override per query).
    pub ef_search: usize,
    /// RNG seed for level assignment (deterministic builds in tests).
    pub seed: u64,
}

impl Default for HnswConfig {
    fn default() -> Self {
        HnswConfig {
            m: 16,
            ef_construction: 200,
            ef_search: 64,
            seed: 0x5261_6756,
        }
    }
}

/// Min-heap entry by distance (smaller distance = better).
#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f32,
    id: u32,
}

impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for Candidate {}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // BinaryHeap is a max-heap; reverse to get closest-first.
        other
            .dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| other.id.cmp(&self.id))
    }
}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Max-heap entry by distance (for bounding the result beam).
#[derive(Debug, Clone, Copy)]
struct FarCandidate {
    dist: f32,
    id: u32,
}
impl PartialEq for FarCandidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist && self.id == other.id
    }
}
impl Eq for FarCandidate {}
impl Ord for FarCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}
impl PartialOrd for FarCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct Node {
    /// adjacency[layer] = neighbor ids. len = level + 1.
    layers: Vec<Vec<u32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Hnsw {
    config: HnswConfig,
    nodes: Vec<Node>,
    entry_point: Option<u32>,
    max_level: usize,
    /// multiplier for random level assignment: 1 / ln(M)
    level_mult: f64,
    #[serde(skip, default = "default_rng_state")]
    rng_state: u64,
}

fn default_rng_state() -> u64 {
    0
}

impl Hnsw {
    pub fn new(config: HnswConfig) -> Self {
        let level_mult = 1.0 / (config.m.max(2) as f64).ln();
        let rng_state = config.seed;
        Hnsw {
            config,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            level_mult,
            rng_state,
        }
    }

    pub fn config(&self) -> &HnswConfig {
        &self.config
    }

    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    fn random_level(&mut self) -> usize {
        // xorshift64* — deterministic given the seed, no allocation.
        let mut x = self.rng_state.max(1);
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.rng_state = x;
        let unit = (x.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64
            / (1u64 << 53) as f64;
        let level = (-unit.max(f64::MIN_POSITIVE).ln() * self.level_mult).floor() as usize;
        level.min(31)
    }

    fn distance(arena: &VectorArena, id: u32, query: &[f32]) -> f32 {
        // Arena scores are "higher is better"; graph wants distances.
        -arena.score(id, query)
    }

    fn max_degree(&self, layer: usize) -> usize {
        if layer == 0 {
            self.config.m * 2
        } else {
            self.config.m
        }
    }

    /// Insert node `id` (must equal `self.nodes.len()`; ids are arena rows
    /// appended in lockstep).
    pub fn insert(&mut self, arena: &VectorArena, id: u32) {
        assert_eq!(
            id as usize,
            self.nodes.len(),
            "hnsw ids must be appended in arena order"
        );
        let level = self.random_level();
        let query = arena.get(id).to_vec();
        self.nodes.push(Node {
            layers: vec![Vec::new(); level + 1],
        });

        let Some(mut current) = self.entry_point else {
            self.entry_point = Some(id);
            self.max_level = level;
            return;
        };

        // Greedy descent through layers above the node's level.
        let mut current_dist = Self::distance(arena, current, &query);
        for layer in (level + 1..=self.max_level).rev() {
            loop {
                let mut improved = false;
                // Clone-free neighbor walk: index into self.nodes.
                let neighbors = &self.nodes[current as usize].layers[layer];
                let mut best = (current, current_dist);
                for &n in neighbors {
                    let d = Self::distance(arena, n, &query);
                    if d < best.1 {
                        best = (n, d);
                        improved = true;
                    }
                }
                current = best.0;
                current_dist = best.1;
                if !improved {
                    break;
                }
            }
        }

        // Beam search + connect on each layer from min(level, max_level) down.
        let ef = self.config.ef_construction;
        let top_layer = level.min(self.max_level);
        let mut entry_points = vec![current];
        for layer in (0..=top_layer).rev() {
            let found = self.search_layer(arena, &query, &entry_points, ef, layer, None);
            let neighbors = self.select_neighbors(arena, &found, self.config.m);
            for &(n, _) in &neighbors {
                self.connect(arena, id, n, layer);
            }
            entry_points = found.iter().map(|c| c.id).collect();
            if entry_points.is_empty() {
                entry_points = vec![current];
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(id);
        }
    }

    /// Bidirectional connect with degree-bounded pruning via the diversity
    /// heuristic.
    fn connect(&mut self, arena: &VectorArena, a: u32, b: u32, layer: usize) {
        if a == b {
            return;
        }
        let max_degree = self.max_degree(layer);
        for (from, to) in [(a, b), (b, a)] {
            let list = &mut self.nodes[from as usize].layers[layer];
            if list.contains(&to) {
                continue;
            }
            list.push(to);
            if list.len() > max_degree {
                // Re-select best neighbors with the diversity heuristic.
                let from_vec = arena.get(from).to_vec();
                let candidates: Vec<Candidate> = self.nodes[from as usize].layers[layer]
                    .iter()
                    .map(|&n| Candidate {
                        dist: Self::distance(arena, n, &from_vec),
                        id: n,
                    })
                    .collect();
                let selected = self.select_neighbors(arena, &candidates, max_degree);
                self.nodes[from as usize].layers[layer] =
                    selected.into_iter().map(|(id, _)| id).collect();
            }
        }
    }

    /// The HNSW neighbor-selection heuristic: keep a candidate only if it is
    /// closer to the query node than to any already-selected neighbor
    /// (promotes diverse directions instead of a tight cluster).
    fn select_neighbors(
        &self,
        arena: &VectorArena,
        candidates: &[Candidate],
        m: usize,
    ) -> Vec<(u32, f32)> {
        let mut sorted: Vec<Candidate> = candidates.to_vec();
        sorted.sort_by(|x, y| x.dist.partial_cmp(&y.dist).unwrap_or(Ordering::Equal));
        sorted.dedup_by_key(|c| c.id);
        let mut selected: Vec<(u32, f32)> = Vec::with_capacity(m);
        for c in &sorted {
            if selected.len() >= m {
                break;
            }
            let c_vec = arena.get(c.id);
            let diverse = selected.iter().all(|&(s, _)| {
                let d_cs = match arena.metric() {
                    ragvault_core::Metric::L2 => crate::kernels::l2_sq(c_vec, arena.get(s)),
                    _ => -crate::kernels::dot(c_vec, arena.get(s)),
                };
                c.dist < d_cs
            });
            if diverse {
                selected.push((c.id, c.dist));
            }
        }
        // Backfill with closest remaining if the heuristic was too strict.
        if selected.len() < m {
            for c in &sorted {
                if selected.len() >= m {
                    break;
                }
                if !selected.iter().any(|&(s, _)| s == c.id) {
                    selected.push((c.id, c.dist));
                }
            }
        }
        selected
    }

    /// Beam search on one layer. `accept` (when provided) gates which nodes
    /// may enter the *result* beam; rejected nodes are still traversed as
    /// bridges. Returns candidates sorted closest-first.
    fn search_layer(
        &self,
        arena: &VectorArena,
        query: &[f32],
        entry_points: &[u32],
        ef: usize,
        layer: usize,
        accept: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<Candidate> {
        let mut visited = vec![false; self.nodes.len()];
        let mut to_visit: BinaryHeap<Candidate> = BinaryHeap::new();
        let mut results: BinaryHeap<FarCandidate> = BinaryHeap::new();

        let admit = |id: u32| -> bool {
            if arena.is_deleted(id) {
                return false;
            }
            accept.map(|f| f(id)).unwrap_or(true)
        };

        for &ep in entry_points {
            if ep as usize >= self.nodes.len() || visited[ep as usize] {
                continue;
            }
            visited[ep as usize] = true;
            let d = Self::distance(arena, ep, query);
            to_visit.push(Candidate { dist: d, id: ep });
            if admit(ep) {
                results.push(FarCandidate { dist: d, id: ep });
            }
        }

        while let Some(closest) = to_visit.pop() {
            let worst = results.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
            if closest.dist > worst && results.len() >= ef {
                break;
            }
            for &neighbor in &self.nodes[closest.id as usize].layers[layer] {
                let slot = &mut visited[neighbor as usize];
                if *slot {
                    continue;
                }
                *slot = true;
                let d = Self::distance(arena, neighbor, query);
                let worst = results.peek().map(|f| f.dist).unwrap_or(f32::INFINITY);
                if d < worst || results.len() < ef {
                    to_visit.push(Candidate { dist: d, id: neighbor });
                    if admit(neighbor) {
                        results.push(FarCandidate { dist: d, id: neighbor });
                        if results.len() > ef {
                            results.pop();
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results
            .into_vec()
            .into_iter()
            .map(|f| Candidate {
                dist: f.dist,
                id: f.id,
            })
            .collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out
    }

    /// Approximate top-k search. Returns (id, score) with arena semantics
    /// (higher score = better). `accept` integrates metadata filtering into
    /// the traversal.
    pub fn search(
        &self,
        arena: &VectorArena,
        prepared_query: &[f32],
        k: usize,
        ef: usize,
        accept: Option<&dyn Fn(u32) -> bool>,
    ) -> Vec<(u32, f32)> {
        let Some(entry) = self.entry_point else {
            return Vec::new();
        };
        if k == 0 {
            return Vec::new();
        }
        let ef = ef.max(k);

        // Greedy descent to layer 1.
        let mut current = entry;
        let mut current_dist = Self::distance(arena, current, prepared_query);
        for layer in (1..=self.max_level).rev() {
            loop {
                let mut improved = false;
                for &n in &self.nodes[current as usize].layers[layer] {
                    let d = Self::distance(arena, n, prepared_query);
                    if d < current_dist {
                        current = n;
                        current_dist = d;
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
        }

        let found = self.search_layer(arena, prepared_query, &[current], ef, 0, accept);
        let mut topk = TopK::new(k);
        for c in found {
            topk.push(c.id, -c.dist);
        }
        topk.into_sorted()
    }

    /// Structural invariant check (used by tests and `doctor`).
    pub fn validate(&self) -> Result<(), String> {
        if let Some(ep) = self.entry_point {
            if ep as usize >= self.nodes.len() {
                return Err(format!("entry point {ep} out of range"));
            }
            if self.nodes[ep as usize].layers.len() <= self.max_level
                && !self.nodes.is_empty()
            {
                return Err("entry point does not reach max level".into());
            }
        } else if !self.nodes.is_empty() {
            return Err("non-empty graph without entry point".into());
        }
        for (id, node) in self.nodes.iter().enumerate() {
            for (layer, neighbors) in node.layers.iter().enumerate() {
                if neighbors.len() > self.max_degree(layer) {
                    return Err(format!(
                        "node {id} layer {layer} exceeds degree bound: {}",
                        neighbors.len()
                    ));
                }
                let mut seen = std::collections::HashSet::new();
                for &n in neighbors {
                    if n as usize == id {
                        return Err(format!("self-loop at node {id} layer {layer}"));
                    }
                    if n as usize >= self.nodes.len() {
                        return Err(format!("dangling neighbor {n} at node {id}"));
                    }
                    if !seen.insert(n) {
                        return Err(format!("duplicate neighbor {n} at node {id}"));
                    }
                    if self.nodes[n as usize].layers.len() <= layer {
                        return Err(format!(
                            "neighbor {n} does not exist on layer {layer}"
                        ));
                    }
                }
            }
        }
        Ok(())
    }
}

/// Deterministic builder used in tests/benchmarks: builds a graph over an
/// arena with a fixed seed.
pub fn build_from_arena(arena: &VectorArena, config: HnswConfig) -> Hnsw {
    let mut hnsw = Hnsw::new(config);
    for id in 0..arena.len() as u32 {
        hnsw.insert(arena, id);
    }
    hnsw
}

/// Sample `count` random query vectors of dimension `dim` (test helper).
pub fn random_vectors(count: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..count)
        .map(|_| (0..dim).map(|_| rng.gen_range(-1.0..1.0)).collect())
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use ragvault_core::Metric;

    fn build(n: usize, dim: usize, seed: u64) -> (VectorArena, Hnsw) {
        let mut arena = VectorArena::new(dim, Metric::Cosine);
        for v in random_vectors(n, dim, seed) {
            arena.push(&v).unwrap();
        }
        let hnsw = build_from_arena(&arena, HnswConfig::default());
        (arena, hnsw)
    }

    #[test]
    fn invariants_hold_after_build() {
        let (_, hnsw) = build(2000, 32, 7);
        hnsw.validate().expect("graph invariants");
    }

    #[test]
    fn recall_against_flat_ground_truth() {
        let (arena, hnsw) = build(2000, 64, 42);
        let queries = random_vectors(50, 64, 999);
        let k = 10;
        let mut hits = 0usize;
        let mut total = 0usize;
        for q in &queries {
            let prepared = arena.prepare_query(q).unwrap();
            let truth: Vec<u32> = FlatIndex::search(&arena, &prepared, k, &|_| true)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let approx: Vec<u32> = hnsw
                .search(&arena, &prepared, k, 128, None)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            total += truth.len();
            hits += truth.iter().filter(|id| approx.contains(id)).count();
        }
        let recall = hits as f64 / total as f64;
        assert!(
            recall >= 0.9,
            "recall@10 vs flat ground truth too low: {recall}"
        );
    }

    #[test]
    fn deterministic_across_builds_with_same_seed() {
        let (arena1, hnsw1) = build(500, 16, 3);
        let (_, hnsw2) = build(500, 16, 3);
        let q = arena1.prepare_query(arena1.get(10)).unwrap();
        assert_eq!(
            hnsw1.search(&arena1, &q, 5, 64, None),
            hnsw2.search(&arena1, &q, 5, 64, None)
        );
    }

    #[test]
    fn deleted_nodes_bridge_but_never_appear() {
        let (mut arena, hnsw) = build(500, 16, 5);
        let q = arena.prepare_query(arena.get(42)).unwrap();
        arena.delete(42);
        let results = hnsw.search(&arena, &q, 20, 64, None);
        assert!(results.iter().all(|(id, _)| *id != 42));
        assert_eq!(results.len(), 20);
    }

    #[test]
    fn filtered_search_returns_only_accepted() {
        let (arena, hnsw) = build(1000, 16, 11);
        let q = arena.prepare_query(arena.get(0)).unwrap();
        let accept = |id: u32| id % 5 == 0;
        let results = hnsw.search(&arena, &q, 10, 256, Some(&accept));
        assert!(!results.is_empty());
        assert!(results.iter().all(|(id, _)| id % 5 == 0));
    }

    #[test]
    fn serialization_round_trip_preserves_results() {
        let (arena, hnsw) = build(800, 24, 21);
        let serialized = serde_json::to_vec(&hnsw).unwrap();
        let restored: Hnsw = serde_json::from_slice(&serialized).unwrap();
        restored.validate().expect("restored graph invariants");
        let q = arena.prepare_query(arena.get(1)).unwrap();
        assert_eq!(
            hnsw.search(&arena, &q, 10, 64, None),
            restored.search(&arena, &q, 10, 64, None)
        );
    }

    #[test]
    fn incremental_insert_keeps_invariants() {
        let mut arena = VectorArena::new(8, Metric::Cosine);
        let mut hnsw = Hnsw::new(HnswConfig {
            m: 4,
            ef_construction: 32,
            ef_search: 16,
            seed: 1,
        });
        for (i, v) in random_vectors(300, 8, 2).into_iter().enumerate() {
            let id = arena.push(&v).unwrap();
            hnsw.insert(&arena, id);
            if i % 97 == 0 {
                hnsw.validate().expect("invariants during incremental build");
            }
        }
        hnsw.validate().expect("final invariants");
    }
}
