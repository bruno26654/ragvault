//! IVF (inverted file) index with optional Product Quantization.
//!
//! - Training: Lloyd k-means over an L2 space on a bounded sample
//!   (deterministic given the seed). L2 is used for partitioning under every
//!   metric; candidate scoring always goes through the metric-aware arena
//!   (IVF-Flat) or ADC + f32 rescore (IVF-PQ), so the final ranking respects
//!   the collection's metric.
//! - Search: probe the `nprobe` closest lists; rows appended after the last
//!   build (`built_rows..`) are brute-force scanned and merged, so the index
//!   never misses fresh writes between rebuilds.
//! - PQ: `m` contiguous subspaces, 256 centroids each (8-bit codes), ADC
//!   lookup tables per query, oversampled candidates rescored against the
//!   f32 arena. Both dot and L2 decompose exactly across subspaces, so the
//!   ADC approximation is unbiased up to quantization error.
//! - The index is a rebuildable acceleration structure: it is NOT
//!   persisted; the engine rebuilds it from the arena on open/flush/compact
//!   (documented in docs/RETRIEVAL.md).

use ragvault_core::{Error, Metric, Result};

use crate::arena::VectorArena;
use crate::kernels;
use crate::topk::TopK;

#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct IvfConfig {
    /// 0 = auto: sqrt(n) clamped to [16, 1024].
    #[serde(default)]
    pub nlist: usize,
    #[serde(default = "default_nprobe")]
    pub nprobe: usize,
    /// PQ subspace count; 0 disables PQ (IVF-Flat). Must divide the
    /// dimension when set.
    #[serde(default)]
    pub pq_m: usize,
    #[serde(default = "default_seed")]
    pub seed: u64,
}

fn default_nprobe() -> usize {
    8
}
fn default_seed() -> u64 {
    0x4956_4621
}

impl Default for IvfConfig {
    fn default() -> Self {
        IvfConfig {
            nlist: 0,
            nprobe: default_nprobe(),
            pq_m: 0,
            seed: default_seed(),
        }
    }
}

/// Minimum rows before training an IVF is worthwhile.
pub const MIN_TRAIN_ROWS: usize = 256;
const KMEANS_ITERS: usize = 8;
const TRAIN_SAMPLE: usize = 20_000;
const PQ_CENTROIDS: usize = 256;

struct Pq {
    m: usize,
    sub_dim: usize,
    /// centroids[s] is a (k_s * sub_dim) row-major table.
    centroids: Vec<Vec<f32>>,
    /// codes[row * m + s]
    codes: Vec<u8>,
}

pub struct IvfIndex {
    dim: usize,
    metric: Metric,
    nlist: usize,
    /// nlist * dim row-major.
    centroids: Vec<f32>,
    lists: Vec<Vec<u32>>,
    pq: Option<Pq>,
    /// Rows covered by this build; rows >= built_rows are delta-scanned.
    built_rows: u32,
}

fn xorshift(state: &mut u64) -> u64 {
    let mut x = state.wrapping_mul(2685821657736338717).max(1);
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

/// Deterministic sample of up to `take` distinct values from 0..n.
fn sample_rows(n: usize, take: usize, seed: u64) -> Vec<u32> {
    if n <= take {
        return (0..n as u32).collect();
    }
    // Reservoir sampling with a seeded xorshift.
    let mut reservoir: Vec<u32> = (0..take as u32).collect();
    let mut state = seed;
    for i in take..n {
        let j = (xorshift(&mut state) % (i as u64 + 1)) as usize;
        if j < take {
            reservoir[j] = i as u32;
        }
    }
    reservoir
}

/// Lloyd k-means in L2 over `rows` of `data` (each `dim` wide).
/// Returns `k * dim` centroids. Deterministic given the seed.
fn kmeans<'a, F: Fn(u32) -> &'a [f32]>(
    get: &F,
    rows: &[u32],
    dim: usize,
    k: usize,
    seed: u64,
) -> Vec<f32> {
    let k = k.min(rows.len()).max(1);
    let mut state = seed;
    // Init: k distinct sampled rows.
    let mut centroids = Vec::with_capacity(k * dim);
    let init = sample_rows(rows.len(), k, xorshift(&mut state));
    for &idx in &init {
        centroids.extend_from_slice(get(rows[idx as usize]));
    }
    let mut assignment = vec![0usize; rows.len()];
    for _iter in 0..KMEANS_ITERS {
        // Assign.
        for (i, &row) in rows.iter().enumerate() {
            let v = get(row);
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d = kernels::l2_sq(v, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            assignment[i] = best;
        }
        // Update.
        let mut sums = vec![0.0f64; k * dim];
        let mut counts = vec![0u32; k];
        for (i, &row) in rows.iter().enumerate() {
            let c = assignment[i];
            counts[c] += 1;
            for (d, &x) in get(row).iter().enumerate() {
                sums[c * dim + d] += f64::from(x);
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Re-seed empty centroid from a random row.
                let r = rows[(xorshift(&mut state) % rows.len() as u64) as usize];
                centroids[c * dim..(c + 1) * dim].copy_from_slice(get(r));
            } else {
                for d in 0..dim {
                    centroids[c * dim + d] = (sums[c * dim + d] / f64::from(counts[c])) as f32;
                }
            }
        }
    }
    centroids
}

impl IvfIndex {
    /// Train and build over all rows currently in the arena (tombstoned rows
    /// are indexed but filtered at query time, like every other backend).
    pub fn build(arena: &VectorArena, config: &IvfConfig) -> Result<IvfIndex> {
        let n = arena.len();
        let dim = arena.dim();
        if n < MIN_TRAIN_ROWS {
            return Err(Error::invalid(
                "ivf",
                format!("at least {MIN_TRAIN_ROWS} rows to train"),
                format!("{n}"),
            ));
        }
        if config.pq_m > 0 && !dim.is_multiple_of(config.pq_m) {
            return Err(Error::invalid(
                "ivf.pq_m",
                format!("a divisor of dim {dim}"),
                format!("{}", config.pq_m),
            ));
        }
        let nlist = if config.nlist == 0 {
            ((n as f64).sqrt() as usize).clamp(16, 1024)
        } else {
            config.nlist
        };
        let sample = sample_rows(n, TRAIN_SAMPLE, config.seed);
        let get = |row: u32| arena.get(row);
        let centroids = kmeans(&get, &sample, dim, nlist, config.seed);

        // Assign every row to its closest centroid.
        let mut lists: Vec<Vec<u32>> = vec![Vec::new(); nlist];
        for row in 0..n as u32 {
            let v = arena.get(row);
            let mut best = 0usize;
            let mut best_d = f32::INFINITY;
            for c in 0..nlist {
                let d = kernels::l2_sq(v, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c;
                }
            }
            lists[best].push(row);
        }

        let pq = if config.pq_m > 0 {
            Some(Self::train_pq(arena, &sample, config.pq_m, config.seed)?)
        } else {
            None
        };

        Ok(IvfIndex {
            dim,
            metric: arena.metric(),
            nlist,
            centroids,
            lists,
            pq,
            built_rows: n as u32,
        })
    }

    fn train_pq(arena: &VectorArena, sample: &[u32], m: usize, seed: u64) -> Result<Pq> {
        let dim = arena.dim();
        let sub_dim = dim / m;
        let mut centroids = Vec::with_capacity(m);
        for s in 0..m {
            let offset = s * sub_dim;
            let get_sub = move |row: u32| &arena.get(row)[offset..offset + sub_dim];
            let k = PQ_CENTROIDS.min(sample.len());
            let table = kmeans(&get_sub, sample, sub_dim, k, seed.wrapping_add(s as u64));
            centroids.push(table);
        }
        // Encode all rows.
        let n = arena.len();
        let mut codes = vec![0u8; n * m];
        for row in 0..n as u32 {
            let v = arena.get(row);
            for (s, table) in centroids.iter().enumerate() {
                let sub = &v[s * sub_dim..(s + 1) * sub_dim];
                let k = table.len() / sub_dim;
                let mut best = 0usize;
                let mut best_d = f32::INFINITY;
                for c in 0..k {
                    let d = kernels::l2_sq(sub, &table[c * sub_dim..(c + 1) * sub_dim]);
                    if d < best_d {
                        best_d = d;
                        best = c;
                    }
                }
                codes[row as usize * m + s] = best as u8;
            }
        }
        Ok(Pq {
            m,
            sub_dim,
            centroids,
            codes,
        })
    }

    pub fn built_rows(&self) -> u32 {
        self.built_rows
    }

    pub fn nlist(&self) -> usize {
        self.nlist
    }

    pub fn uses_pq(&self) -> bool {
        self.pq.is_some()
    }

    /// Approximate memory of the acceleration structures (stats/doctor).
    pub fn memory_bytes(&self) -> usize {
        let lists: usize = self.lists.iter().map(|l| l.len() * 4).sum();
        let pq = self
            .pq
            .as_ref()
            .map(|p| p.codes.len() + p.centroids.iter().map(|c| c.len() * 4).sum::<usize>())
            .unwrap_or(0);
        self.centroids.len() * 4 + lists + pq
    }

    /// ADC lookup tables for a prepared query: `tables[s][c]` = partial score
    /// of centroid `c` in subspace `s` (dot for cosine/dot, negated L2 for
    /// L2 — both sum across subspaces to the full approximate score).
    fn adc_tables(&self, pq: &Pq, query: &[f32]) -> Vec<Vec<f32>> {
        let mut tables = Vec::with_capacity(pq.m);
        for (s, table) in pq.centroids.iter().enumerate() {
            let sub_q = &query[s * pq.sub_dim..(s + 1) * pq.sub_dim];
            let k = table.len() / pq.sub_dim;
            let mut row = Vec::with_capacity(k);
            for c in 0..k {
                let cent = &table[c * pq.sub_dim..(c + 1) * pq.sub_dim];
                let score = match self.metric {
                    Metric::L2 => -kernels::l2_sq(sub_q, cent),
                    _ => kernels::dot(sub_q, cent),
                };
                row.push(score);
            }
            tables.push(row);
        }
        tables
    }

    /// Search: probe `nprobe` closest lists over built rows, brute-force the
    /// delta (rows appended after the build), merge. With PQ, candidates are
    /// ADC-scored with `oversample`x pool then rescored against f32.
    pub fn search(
        &self,
        arena: &VectorArena,
        prepared_query: &[f32],
        k: usize,
        nprobe: usize,
        accept: &dyn Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        if k == 0 {
            return Vec::new();
        }
        let nprobe = nprobe.clamp(1, self.nlist);
        // Rank lists by centroid distance.
        let mut order = TopK::new(nprobe);
        for c in 0..self.nlist {
            let d = kernels::l2_sq(
                prepared_query,
                &self.centroids[c * self.dim..(c + 1) * self.dim],
            );
            order.push(c as u32, -d);
        }

        let mut results = TopK::new(k);
        match &self.pq {
            None => {
                for (list, _) in order.into_sorted() {
                    for &row in &self.lists[list as usize] {
                        if !arena.is_deleted(row) && accept(row) {
                            results.push(row, arena.score(row, prepared_query));
                        }
                    }
                }
            }
            Some(pq) => {
                let tables = self.adc_tables(pq, prepared_query);
                let oversample = (k * 8).max(k + 32);
                let mut approx = TopK::new(oversample);
                for (list, _) in order.into_sorted() {
                    for &row in &self.lists[list as usize] {
                        if arena.is_deleted(row) || !accept(row) {
                            continue;
                        }
                        let base = row as usize * pq.m;
                        let mut score = 0.0f32;
                        for (s, table) in tables.iter().enumerate() {
                            score += table[pq.codes[base + s] as usize];
                        }
                        approx.push(row, score);
                    }
                }
                for (row, _) in approx.into_sorted() {
                    results.push(row, arena.score(row, prepared_query));
                }
            }
        }
        // Delta rows written after the last build: exact scan, always seen.
        for row in self.built_rows..arena.len() as u32 {
            if !arena.is_deleted(row) && accept(row) {
                results.push(row, arena.score(row, prepared_query));
            }
        }
        results.into_sorted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::flat::FlatIndex;
    use crate::hnsw::random_vectors;

    fn build_arena(n: usize, dim: usize, seed: u64) -> VectorArena {
        let mut arena = VectorArena::new(dim, Metric::Cosine);
        for v in random_vectors(n, dim, seed) {
            arena.push(&v).unwrap();
        }
        arena
    }

    fn recall_vs_flat(
        arena: &VectorArena,
        ivf: &IvfIndex,
        nprobe: usize,
        n_queries: usize,
        k: usize,
    ) -> f64 {
        let queries = random_vectors(n_queries, arena.dim(), 4242);
        let mut hits = 0;
        let mut total = 0;
        for q in &queries {
            let prepared = arena.prepare_query(q).unwrap();
            let truth: Vec<u32> = FlatIndex::search(arena, &prepared, k, &|_| true)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let got: Vec<u32> = ivf
                .search(arena, &prepared, k, nprobe, &|_| true)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            total += truth.len();
            hits += truth.iter().filter(|id| got.contains(id)).count();
        }
        hits as f64 / total as f64
    }

    #[test]
    fn full_probe_equals_flat() {
        let arena = build_arena(2000, 32, 1);
        let ivf = IvfIndex::build(&arena, &IvfConfig::default()).unwrap();
        let recall = recall_vs_flat(&arena, &ivf, ivf.nlist(), 20, 10);
        assert_eq!(recall, 1.0, "probing every list must be exact");
    }

    #[test]
    fn recall_grows_with_nprobe() {
        let arena = build_arena(4000, 32, 2);
        let ivf = IvfIndex::build(&arena, &IvfConfig::default()).unwrap();
        let low = recall_vs_flat(&arena, &ivf, 1, 30, 10);
        let mid = recall_vs_flat(&arena, &ivf, 8, 30, 10);
        let high = recall_vs_flat(&arena, &ivf, 24, 30, 10);
        assert!(mid >= low, "recall must not shrink with more probes");
        assert!(high >= mid);
        assert!(high >= 0.85, "nprobe=24/63 should reach >=0.85, got {high}");
    }

    #[test]
    fn pq_with_rescore_reaches_high_recall() {
        let arena = build_arena(3000, 64, 3);
        let config = IvfConfig {
            pq_m: 8,
            ..Default::default()
        };
        let ivf = IvfIndex::build(&arena, &config).unwrap();
        assert!(ivf.uses_pq());
        let recall = recall_vs_flat(&arena, &ivf, 24, 30, 10);
        assert!(recall >= 0.8, "ivf-pq recall too low: {recall}");
        // full probe with rescore should stay close to exact even with PQ
        let full = recall_vs_flat(&arena, &ivf, ivf.nlist(), 30, 10);
        assert!(full >= 0.9, "ivf-pq full-probe recall too low: {full}");
        // PQ codes must be ~m bytes/vector plus tables.
        assert!(ivf.memory_bytes() < 3000 * 64 * 4 / 4);
    }

    #[test]
    fn delta_rows_are_always_found() {
        let mut arena = build_arena(1000, 16, 5);
        let ivf = IvfIndex::build(&arena, &IvfConfig::default()).unwrap();
        // Append a fresh row AFTER the build; it must be findable.
        let fresh = random_vectors(1, 16, 999).pop().unwrap();
        let fresh_row = arena.push(&fresh).unwrap();
        let prepared = arena.prepare_query(&fresh).unwrap();
        let results = ivf.search(&arena, &prepared, 3, 1, &|_| true);
        assert_eq!(
            results[0].0, fresh_row,
            "delta row must rank first for itself"
        );
    }

    #[test]
    fn tombstones_and_filters_are_respected() {
        let mut arena = build_arena(1000, 16, 7);
        let ivf = IvfIndex::build(&arena, &IvfConfig::default()).unwrap();
        let q = arena.prepare_query(arena.get(10)).unwrap().to_vec();
        arena.delete(10);
        let results = ivf.search(&arena, &q, 20, ivf.nlist(), &|_| true);
        assert!(results.iter().all(|(id, _)| *id != 10));
        let filtered = ivf.search(&arena, &q, 20, ivf.nlist(), &|id| id % 2 == 0);
        assert!(!filtered.is_empty());
        assert!(filtered.iter().all(|(id, _)| id % 2 == 0));
    }

    #[test]
    fn deterministic_given_seed() {
        let arena = build_arena(1500, 16, 9);
        let a = IvfIndex::build(&arena, &IvfConfig::default()).unwrap();
        let b = IvfIndex::build(&arena, &IvfConfig::default()).unwrap();
        let q = arena.prepare_query(arena.get(3)).unwrap();
        assert_eq!(
            a.search(&arena, &q, 10, 8, &|_| true),
            b.search(&arena, &q, 10, 8, &|_| true)
        );
    }

    #[test]
    fn rejects_bad_configs() {
        let arena = build_arena(300, 30, 11);
        let bad_pq = IvfConfig {
            pq_m: 7, // 30 % 7 != 0
            ..Default::default()
        };
        assert!(IvfIndex::build(&arena, &bad_pq).is_err());
        let tiny = build_arena(10, 8, 12);
        assert!(IvfIndex::build(&tiny, &IvfConfig::default()).is_err());
    }
}
