//! SQ8 scalar quantization: 4x smaller vectors, int8 dot-product scans.
//!
//! Each vector is quantized independently: `scale = max_abs / 127`,
//! `code_i = round(v_i / scale)`. The approximate dot product of a stored
//! row against a quantized query is `row_scale * query_scale * Σ(ci * qi)`
//! with i32 accumulation (no overflow: 127*127*dim fits i32 for dim < 2^15).
//!
//! Search flow (engine's `sq8_flat` backend): scan int8 codes with an
//! oversampled pool, rescore the survivors against the original f32 arena,
//! then take the final top-k. Rescoring makes end recall near-exact while
//! the scan touches 4x less memory than f32.
//!
//! v0.1 scope: cosine/dot metrics (the engine gates it); L2 stays on f32.

use ragvault_core::{Error, Metric, Result};

use crate::topk::TopK;

/// A quantized vector: per-vector scale + i8 codes.
#[derive(Debug, Clone)]
pub struct QuantizedQuery {
    pub scale: f32,
    pub codes: Vec<i8>,
}

#[derive(Debug, Clone, Default)]
pub struct Sq8Arena {
    dim: usize,
    scales: Vec<f32>,
    codes: Vec<i8>,
}

/// Unrolled i8 dot product with i32 accumulation (auto-vectorized).
pub fn dot_i8(a: &[i8], b: &[i8]) -> i32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0i32; 4];
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        acc[0] += i32::from(a[base]) * i32::from(b[base]);
        acc[1] += i32::from(a[base + 1]) * i32::from(b[base + 1]);
        acc[2] += i32::from(a[base + 2]) * i32::from(b[base + 2]);
        acc[3] += i32::from(a[base + 3]) * i32::from(b[base + 3]);
    }
    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for i in chunks * 4..a.len() {
        sum += i32::from(a[i]) * i32::from(b[i]);
    }
    sum
}

/// Scalar reference for differential tests.
pub fn dot_i8_ref(a: &[i8], b: &[i8]) -> i32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| i32::from(*x) * i32::from(*y))
        .sum()
}

pub fn quantize(vector: &[f32]) -> (f32, Vec<i8>) {
    let max_abs = vector.iter().fold(0.0f32, |m, x| m.max(x.abs()));
    if max_abs == 0.0 || !max_abs.is_finite() {
        return (0.0, vec![0; vector.len()]);
    }
    let scale = max_abs / 127.0;
    let inv = 1.0 / scale;
    let codes = vector
        .iter()
        .map(|x| (x * inv).round().clamp(-127.0, 127.0) as i8)
        .collect();
    (scale, codes)
}

impl Sq8Arena {
    pub fn new(dim: usize, metric: Metric) -> Result<Self> {
        if metric == Metric::L2 {
            return Err(Error::invalid(
                "quantization",
                "cosine or dot metric for sq8 (L2 is not supported in v0.1)",
                "l2",
            ));
        }
        Ok(Sq8Arena {
            dim,
            scales: Vec::new(),
            codes: Vec::new(),
        })
    }

    pub fn len(&self) -> usize {
        self.scales.len()
    }

    pub fn is_empty(&self) -> bool {
        self.scales.is_empty()
    }

    /// Bytes used by codes + scales (for stats; excludes Vec overhead).
    pub fn memory_bytes(&self) -> usize {
        self.codes.len() + self.scales.len() * 4
    }

    /// Append a vector (must be called in lockstep with the f32 arena; the
    /// caller passes the already-normalized row for cosine).
    pub fn push(&mut self, vector: &[f32]) -> Result<()> {
        if vector.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        let (scale, codes) = quantize(vector);
        self.scales.push(scale);
        self.codes.extend_from_slice(&codes);
        Ok(())
    }

    pub fn prepare_query(&self, prepared_f32_query: &[f32]) -> Result<QuantizedQuery> {
        if prepared_f32_query.len() != self.dim {
            return Err(Error::DimensionMismatch {
                expected: self.dim,
                got: prepared_f32_query.len(),
            });
        }
        let (scale, codes) = quantize(prepared_f32_query);
        Ok(QuantizedQuery { scale, codes })
    }

    #[inline]
    pub fn score(&self, id: u32, query: &QuantizedQuery) -> f32 {
        let start = id as usize * self.dim;
        let row = &self.codes[start..start + self.dim];
        self.scales[id as usize] * query.scale * dot_i8(row, &query.codes) as f32
    }

    /// Approximate top-`pool` scan over quantized codes with an integrated
    /// accept predicate (true prefilter: rejected rows are never scored).
    pub fn scan(
        &self,
        query: &QuantizedQuery,
        pool: usize,
        accept: &dyn Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        let mut topk = TopK::new(pool);
        for id in 0..self.len() as u32 {
            if accept(id) {
                topk.push(id, self.score(id, query));
            }
        }
        topk.into_sorted()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::arena::VectorArena;
    use crate::flat::FlatIndex;
    use crate::hnsw::random_vectors;

    #[test]
    fn dot_i8_matches_reference() {
        for dim in [1, 3, 16, 384, 1000] {
            let a: Vec<i8> = (0..dim).map(|i| ((i * 37 + 11) % 255) as i8).collect();
            let b: Vec<i8> = (0..dim).map(|i| ((i * 91 + 3) % 255) as i8).collect();
            assert_eq!(dot_i8(&a, &b), dot_i8_ref(&a, &b), "dim={dim}");
        }
    }

    #[test]
    fn quantized_dot_approximates_f32_dot() {
        for seed in 1..5u64 {
            let vecs = random_vectors(2, 384, seed);
            let (a, b) = (&vecs[0], &vecs[1]);
            let exact = crate::kernels::dot(a, b);
            let (sa, ca) = quantize(a);
            let (sb, cb) = quantize(b);
            let approx = sa * sb * dot_i8(&ca, &cb) as f32;
            // Norms here are O(sqrt(dim)); allow 1% of the max possible dot.
            let bound = 0.01 * crate::kernels::norm(a) * crate::kernels::norm(b);
            assert!(
                (exact - approx).abs() <= bound,
                "seed={seed}: exact={exact} approx={approx}"
            );
        }
    }

    #[test]
    fn zero_vector_is_safe() {
        let (scale, codes) = quantize(&[0.0; 8]);
        assert_eq!(scale, 0.0);
        assert!(codes.iter().all(|&c| c == 0));
    }

    #[test]
    fn scan_with_rescore_matches_exact_topk() {
        // End-to-end recall check: sq8 scan (4x pool) + f32 rescore == flat.
        let dim = 128;
        let n = 3000;
        let k = 10;
        let mut arena = VectorArena::new(dim, Metric::Cosine);
        let mut sq8 = Sq8Arena::new(dim, Metric::Cosine).unwrap();
        for v in random_vectors(n, dim, 77) {
            let id = arena.push(&v).unwrap();
            sq8.push(arena.get(id)).unwrap();
        }
        let queries = random_vectors(20, dim, 999);
        let mut hits = 0;
        let mut total = 0;
        for q in &queries {
            let prepared = arena.prepare_query(q).unwrap();
            let exact: Vec<u32> = FlatIndex::search(&arena, &prepared, k, &|_| true)
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            let qq = sq8.prepare_query(&prepared).unwrap();
            let candidates = sq8.scan(&qq, k * 4, &|_| true);
            let mut rescored = TopK::new(k);
            for (id, _) in candidates {
                rescored.push(id, arena.score(id, &prepared));
            }
            let approx: Vec<u32> = rescored
                .into_sorted()
                .into_iter()
                .map(|(id, _)| id)
                .collect();
            total += exact.len();
            hits += exact.iter().filter(|id| approx.contains(id)).count();
        }
        let recall = hits as f64 / total as f64;
        assert!(recall >= 0.99, "sq8+rescore recall too low: {recall}");
    }

    #[test]
    fn l2_metric_rejected() {
        assert!(Sq8Arena::new(8, Metric::L2).is_err());
    }

    #[test]
    fn memory_is_about_a_quarter_of_f32() {
        let dim = 256;
        let mut sq8 = Sq8Arena::new(dim, Metric::Cosine).unwrap();
        for v in random_vectors(100, dim, 5) {
            sq8.push(&v).unwrap();
        }
        let f32_bytes = 100 * dim * 4;
        assert!(
            sq8.memory_bytes() < f32_bytes / 3,
            "expected ~4x compression"
        );
    }
}
