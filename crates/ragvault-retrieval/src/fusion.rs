//! Candidate fusion.
//!
//! Reciprocal Rank Fusion (RRF) with per-signal weights is the default: it is
//! rank-based, so it never sums incompatible score scales (BM25 scores and
//! cosine similarities are not commensurable). The fused score is
//! `sum_s( w_s / (k0 + rank_s) )` — documented semantics, deterministic
//! given inputs.

/// One ranked candidate list from a retrieval signal.
pub struct FusionInput<'a> {
    /// Signal name ("dense", "bm25", "sparse", ...).
    pub name: &'a str,
    pub weight: f32,
    /// (id, raw score) best-first.
    pub ranked: &'a [(u32, f32)],
}

/// Weighted RRF. Returns (id, fused_score) best-first, deterministic
/// (ties broken by id).
pub fn rrf_fuse(inputs: &[FusionInput<'_>], k0: f32, limit: usize) -> Vec<(u32, f32)> {
    let mut scores: std::collections::HashMap<u32, f32> = std::collections::HashMap::new();
    for input in inputs {
        for (rank, (id, _)) in input.ranked.iter().enumerate() {
            let contribution = input.weight / (k0 + rank as f32 + 1.0);
            *scores.entry(*id).or_insert(0.0) += contribution;
        }
    }
    let mut fused: Vec<(u32, f32)> = scores.into_iter().collect();
    fused.sort_by(|a, b| {
        b.1.partial_cmp(&a.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.0.cmp(&b.0))
    });
    fused.truncate(limit);
    fused
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn document_in_both_lists_wins() {
        let dense = vec![(1u32, 0.9f32), (2, 0.8), (3, 0.7)];
        let bm25 = vec![(2u32, 12.0f32), (4, 8.0)];
        let fused = rrf_fuse(
            &[
                FusionInput {
                    name: "dense",
                    weight: 1.0,
                    ranked: &dense,
                },
                FusionInput {
                    name: "bm25",
                    weight: 1.0,
                    ranked: &bm25,
                },
            ],
            60.0,
            10,
        );
        assert_eq!(fused[0].0, 2, "doc present in both signals should lead");
    }

    #[test]
    fn weights_shift_the_ranking() {
        let a = vec![(1u32, 1.0f32)];
        let b = vec![(2u32, 1.0f32)];
        let fused = rrf_fuse(
            &[
                FusionInput {
                    name: "a",
                    weight: 0.1,
                    ranked: &a,
                },
                FusionInput {
                    name: "b",
                    weight: 2.0,
                    ranked: &b,
                },
            ],
            60.0,
            10,
        );
        assert_eq!(fused[0].0, 2);
    }

    #[test]
    fn deterministic_tie_break() {
        let a = vec![(5u32, 1.0f32)];
        let b = vec![(3u32, 1.0f32)];
        let fused = rrf_fuse(
            &[
                FusionInput {
                    name: "a",
                    weight: 1.0,
                    ranked: &a,
                },
                FusionInput {
                    name: "b",
                    weight: 1.0,
                    ranked: &b,
                },
            ],
            60.0,
            10,
        );
        assert_eq!(fused[0].0, 3, "equal scores tie-break by lower id");
    }

    #[test]
    fn respects_limit() {
        let ranked: Vec<(u32, f32)> = (0..100).map(|i| (i, 100.0 - i as f32)).collect();
        let fused = rrf_fuse(
            &[FusionInput {
                name: "dense",
                weight: 1.0,
                ranked: &ranked,
            }],
            60.0,
            7,
        );
        assert_eq!(fused.len(), 7);
    }
}
