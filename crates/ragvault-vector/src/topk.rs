//! Bounded top-k selection.
//!
//! A size-capped binary min-heap: streaming candidates never sort the whole
//! result set, and extracting the final ranking is `O(k log k)`. For the k
//! ranges RAG uses (k <= a few hundred) this beats full sorts by a wide
//! margin and needs no allocation beyond the k slots.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

/// Candidate ordered so that the *worst* (lowest score) sits at the top of
/// the `BinaryHeap` (max-heap turned min-heap via reversed ordering).
#[derive(Debug, Clone, Copy, PartialEq)]
struct HeapItem {
    score: f32,
    id: u32,
}

impl Eq for HeapItem {}

impl Ord for HeapItem {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse on score so the heap keeps the minimum on top.
        other
            .score
            .partial_cmp(&self.score)
            .unwrap_or(Ordering::Equal)
            // Tie-break on id for deterministic results.
            .then_with(|| other.id.cmp(&self.id))
    }
}

impl PartialOrd for HeapItem {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Bounded top-k accumulator (higher score = better).
#[derive(Debug, Clone)]
pub struct TopK {
    k: usize,
    heap: BinaryHeap<HeapItem>,
}

impl TopK {
    pub fn new(k: usize) -> Self {
        TopK {
            k,
            heap: BinaryHeap::with_capacity(k + 1),
        }
    }

    #[inline]
    pub fn push(&mut self, id: u32, score: f32) {
        if score.is_nan() || self.k == 0 {
            return;
        }
        if self.heap.len() < self.k {
            self.heap.push(HeapItem { score, id });
        } else if let Some(worst) = self.heap.peek() {
            if score > worst.score || (score == worst.score && id < worst.id) {
                self.heap.pop();
                self.heap.push(HeapItem { score, id });
            }
        }
    }

    /// Current admission threshold (score of the worst kept item), if full.
    #[inline]
    pub fn threshold(&self) -> Option<f32> {
        if self.heap.len() >= self.k {
            self.heap.peek().map(|w| w.score)
        } else {
            None
        }
    }

    pub fn len(&self) -> usize {
        self.heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.heap.is_empty()
    }

    /// Extract results sorted best-first.
    pub fn into_sorted(self) -> Vec<(u32, f32)> {
        let mut items: Vec<HeapItem> = self.heap.into_vec();
        items.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.id.cmp(&b.id))
        });
        items.into_iter().map(|i| (i.id, i.score)).collect()
    }

    /// Merge another TopK into this one (used for per-segment merges).
    pub fn merge(&mut self, other: TopK) {
        for item in other.heap.into_vec() {
            self.push(item.id, item.score);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn keeps_best_k_sorted() {
        let mut topk = TopK::new(3);
        for (id, score) in [(1, 0.5), (2, 0.9), (3, 0.1), (4, 0.7), (5, 0.8)] {
            topk.push(id, score);
        }
        assert_eq!(topk.into_sorted(), vec![(2, 0.9), (5, 0.8), (4, 0.7)]);
    }

    #[test]
    fn matches_full_sort_reference() {
        let n = 5000;
        let mut scores: Vec<(u32, f32)> = (0..n)
            .map(|i| (i, ((i as f32 * 37.77).sin() * 1000.0).fract()))
            .collect();
        for k in [1, 10, 32, 100, 500] {
            let mut topk = TopK::new(k);
            for &(id, s) in &scores {
                topk.push(id, s);
            }
            let got = topk.into_sorted();
            scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
            let want: Vec<(u32, f32)> = scores.iter().take(k).copied().collect();
            assert_eq!(got, want, "k={k}");
            scores.sort_by_key(|&(id, _)| id);
        }
    }

    #[test]
    fn nan_scores_are_dropped() {
        let mut topk = TopK::new(2);
        topk.push(1, f32::NAN);
        topk.push(2, 0.5);
        assert_eq!(topk.into_sorted(), vec![(2, 0.5)]);
    }

    #[test]
    fn merge_is_equivalent_to_single_stream() {
        let mut a = TopK::new(4);
        let mut b = TopK::new(4);
        let mut single = TopK::new(4);
        for i in 0..100u32 {
            let score = ((i * 7919) % 101) as f32;
            if i % 2 == 0 {
                a.push(i, score);
            } else {
                b.push(i, score);
            }
            single.push(i, score);
        }
        a.merge(b);
        assert_eq!(a.into_sorted(), single.into_sorted());
    }

    #[test]
    fn zero_k() {
        let mut topk = TopK::new(0);
        topk.push(1, 1.0);
        assert!(topk.into_sorted().is_empty());
    }
}
