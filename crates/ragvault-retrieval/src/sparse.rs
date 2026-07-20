//! Sparse vector index (user-provided sparse embeddings such as SPLADE or
//! BGE-M3 sparse). Inverted lists over dimensions; dot-product scoring.
//! The core never generates sparse embeddings — they are supplied by the
//! caller.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use ragvault_core::{Result, SparseVector};

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct SparseIndex {
    /// dimension -> (doc, value) postings
    postings: HashMap<u32, Vec<(u32, f32)>>,
    len: u32,
}

impl SparseIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn len(&self) -> usize {
        self.len as usize
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Docs must be appended in order; an empty sparse vector is allowed
    /// (the doc simply never matches).
    pub fn add(&mut self, doc: u32, vector: &SparseVector) -> Result<()> {
        assert_eq!(doc, self.len, "sparse ids append-only");
        vector.validate()?;
        for (&idx, &val) in vector.indices.iter().zip(&vector.values) {
            if val != 0.0 {
                self.postings.entry(idx).or_default().push((doc, val));
            }
        }
        self.len += 1;
        Ok(())
    }

    /// Placeholder row for documents without a sparse representation.
    pub fn add_empty(&mut self, doc: u32) {
        assert_eq!(doc, self.len, "sparse ids append-only");
        self.len += 1;
    }

    /// Rebuild the index keeping only docs present in `map` (old id → new
    /// id), used by compaction. `new_len` is the row count of the compacted
    /// arena; every mapped id must be < new_len.
    pub fn remap(&self, map: &HashMap<u32, u32>, new_len: u32) -> SparseIndex {
        let mut postings: HashMap<u32, Vec<(u32, f32)>> = HashMap::new();
        for (&dim, entries) in &self.postings {
            let mut remapped: Vec<(u32, f32)> = entries
                .iter()
                .filter_map(|&(doc, val)| map.get(&doc).map(|&new_doc| (new_doc, val)))
                .collect();
            if !remapped.is_empty() {
                remapped.sort_by_key(|&(doc, _)| doc);
                postings.insert(dim, remapped);
            }
        }
        SparseIndex {
            postings,
            len: new_len,
        }
    }

    /// Sparse dot-product top-k with integrated accept predicate.
    pub fn search(
        &self,
        query: &SparseVector,
        k: usize,
        accept: &dyn Fn(u32) -> bool,
    ) -> Result<Vec<(u32, f32)>> {
        query.validate()?;
        if k == 0 {
            return Ok(Vec::new());
        }
        let mut scores: HashMap<u32, f32> = HashMap::new();
        for (&idx, &qval) in query.indices.iter().zip(&query.values) {
            if let Some(postings) = self.postings.get(&idx) {
                for &(doc, dval) in postings {
                    if accept(doc) {
                        *scores.entry(doc).or_insert(0.0) += qval * dval;
                    }
                }
            }
        }
        let mut results: Vec<(u32, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(k);
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sv(pairs: &[(u32, f32)]) -> SparseVector {
        SparseVector {
            indices: pairs.iter().map(|p| p.0).collect(),
            values: pairs.iter().map(|p| p.1).collect(),
        }
    }

    #[test]
    fn dot_product_ranking() {
        let mut idx = SparseIndex::new();
        idx.add(0, &sv(&[(1, 1.0), (5, 2.0)])).unwrap();
        idx.add(1, &sv(&[(5, 1.0)])).unwrap();
        idx.add_empty(2);
        let results = idx.search(&sv(&[(5, 1.0)]), 10, &|_| true).unwrap();
        assert_eq!(results, vec![(0, 2.0), (1, 1.0)]);
    }

    #[test]
    fn accept_prefilters() {
        let mut idx = SparseIndex::new();
        idx.add(0, &sv(&[(1, 1.0)])).unwrap();
        idx.add(1, &sv(&[(1, 5.0)])).unwrap();
        let results = idx.search(&sv(&[(1, 1.0)]), 10, &|d| d == 0).unwrap();
        assert_eq!(results, vec![(0, 1.0)]);
    }

    #[test]
    fn remap_preserves_scores_for_kept_docs() {
        let mut idx = SparseIndex::new();
        idx.add(0, &sv(&[(1, 1.0)])).unwrap();
        idx.add(1, &sv(&[(1, 5.0), (2, 3.0)])).unwrap();
        idx.add(2, &sv(&[(2, 7.0)])).unwrap();
        // Drop doc 0; doc 1 -> 0, doc 2 -> 1 (as compaction would).
        let map = HashMap::from([(1u32, 0u32), (2u32, 1u32)]);
        let remapped = idx.remap(&map, 2);
        assert_eq!(remapped.len(), 2);
        let results = remapped.search(&sv(&[(2, 1.0)]), 10, &|_| true).unwrap();
        assert_eq!(results, vec![(1, 7.0), (0, 3.0)]);
        let results = remapped.search(&sv(&[(1, 1.0)]), 10, &|_| true).unwrap();
        assert_eq!(results, vec![(0, 5.0)], "dropped doc must vanish");
    }

    #[test]
    fn invalid_vectors_rejected() {
        let mut idx = SparseIndex::new();
        let bad = SparseVector {
            indices: vec![5, 1],
            values: vec![1.0, 1.0],
        };
        assert!(idx.add(0, &bad).is_err());
    }
}
