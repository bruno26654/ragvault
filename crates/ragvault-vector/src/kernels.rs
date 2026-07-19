//! Dense f32 kernels.
//!
//! `dot`, `l2_sq` and `normalize` are the hot-path primitives. The `_ref`
//! variants are the plain scalar references used in differential tests; the
//! main variants use 4-way unrolled accumulators, which LLVM turns into SIMD
//! on every target we ship wheels for, without per-arch intrinsics or
//! `unsafe`. Differential tests pin the two implementations together.

/// Scalar reference dot product.
pub fn dot_ref(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Unrolled dot product (auto-vectorized).
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 4];
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        acc[0] += a[base] * b[base];
        acc[1] += a[base + 1] * b[base + 1];
        acc[2] += a[base + 2] * b[base + 2];
        acc[3] += a[base + 3] * b[base + 3];
    }
    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for i in chunks * 4..a.len() {
        sum += a[i] * b[i];
    }
    sum
}

/// Scalar reference squared L2 distance.
pub fn l2_sq_ref(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum()
}

/// Unrolled squared L2 distance (auto-vectorized).
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 4];
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        let d0 = a[base] - b[base];
        let d1 = a[base + 1] - b[base + 1];
        let d2 = a[base + 2] - b[base + 2];
        let d3 = a[base + 3] - b[base + 3];
        acc[0] += d0 * d0;
        acc[1] += d1 * d1;
        acc[2] += d2 * d2;
        acc[3] += d3 * d3;
    }
    let mut sum = acc[0] + acc[1] + acc[2] + acc[3];
    for i in chunks * 4..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    sum
}

/// L2 norm.
pub fn norm(a: &[f32]) -> f32 {
    dot(a, a).sqrt()
}

/// Normalize in place. Zero vectors are left untouched (documented: a zero
/// vector under cosine scores 0 against everything).
pub fn normalize(a: &mut [f32]) {
    let n = norm(a);
    if n > 0.0 && n.is_finite() {
        let inv = 1.0 / n;
        for x in a.iter_mut() {
            *x *= inv;
        }
    }
}

/// Cosine similarity without pre-normalization (used for reference tests).
pub fn cosine_ref(a: &[f32], b: &[f32]) -> f32 {
    let na = norm(a);
    let nb = norm(b);
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot(a, b) / (na * nb)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pseudo_random_vec(seed: u64, len: usize) -> Vec<f32> {
        // xorshift-based deterministic pseudo-random values in [-1, 1]
        let mut state = seed.wrapping_mul(2685821657736338717).max(1);
        (0..len)
            .map(|_| {
                state ^= state << 13;
                state ^= state >> 7;
                state ^= state << 17;
                ((state % 20_000) as f32 / 10_000.0) - 1.0
            })
            .collect()
    }

    #[test]
    fn dot_matches_reference_across_dims() {
        // priority dims plus deliberately unaligned ones
        for dim in [1, 3, 7, 33, 384, 512, 768, 1000, 1024, 1536, 3072] {
            let a = pseudo_random_vec(dim as u64 + 1, dim);
            let b = pseudo_random_vec(dim as u64 + 7, dim);
            let fast = dot(&a, &b);
            let slow = dot_ref(&a, &b);
            let tolerance = 1e-4 * (dim as f32).sqrt();
            assert!(
                (fast - slow).abs() <= tolerance,
                "dim={dim}: {fast} vs {slow}"
            );
        }
    }

    #[test]
    fn l2_matches_reference_across_dims() {
        for dim in [1, 5, 384, 768, 1023, 1536] {
            let a = pseudo_random_vec(dim as u64 + 3, dim);
            let b = pseudo_random_vec(dim as u64 + 11, dim);
            let fast = l2_sq(&a, &b);
            let slow = l2_sq_ref(&a, &b);
            let tolerance = 1e-4 * (dim as f32).sqrt();
            assert!(
                (fast - slow).abs() <= tolerance,
                "dim={dim}: {fast} vs {slow}"
            );
            assert!(fast >= 0.0, "L2 must be non-negative");
        }
    }

    #[test]
    fn normalize_produces_unit_vectors() {
        let mut v = pseudo_random_vec(42, 384);
        normalize(&mut v);
        assert!((norm(&v) - 1.0).abs() < 1e-5);

        let mut zero = vec![0.0f32; 8];
        normalize(&mut zero);
        assert_eq!(zero, vec![0.0f32; 8]);
    }

    #[test]
    fn cosine_of_normalized_equals_dot() {
        let mut a = pseudo_random_vec(1, 256);
        let mut b = pseudo_random_vec(2, 256);
        let reference = cosine_ref(&a, &b);
        normalize(&mut a);
        normalize(&mut b);
        assert!((dot(&a, &b) - reference).abs() < 1e-5);
    }
}
