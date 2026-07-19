//! Incremental BM25 inverted index.
//!
//! - Unicode-aware tokenizer: lowercased alphanumeric runs (configurable
//!   lowercasing). Works for Latin scripts and CJK-adjacent text degrades to
//!   character runs rather than breaking.
//! - Postings are append-only per term; deletions are handled at query time
//!   through the caller's tombstone predicate, and statistics are kept
//!   consistent by tracking removed document lengths.
//! - Serializable for snapshot persistence.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Bm25Params {
    pub k1: f32,
    pub b: f32,
    pub lowercase: bool,
}

impl Default for Bm25Params {
    fn default() -> Self {
        Bm25Params {
            k1: 1.2,
            b: 0.75,
            lowercase: true,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
struct Posting {
    doc: u32,
    tf: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bm25Index {
    params: Bm25Params,
    postings: HashMap<String, Vec<Posting>>,
    /// token count per internal doc id; 0 means never indexed.
    doc_lengths: Vec<u32>,
    /// doc ids removed logically (their lengths excluded from averages).
    removed: Vec<bool>,
    total_len: u64,
    active_docs: u64,
}

impl Bm25Index {
    pub fn new(params: Bm25Params) -> Self {
        Bm25Index {
            params,
            postings: HashMap::new(),
            doc_lengths: Vec::new(),
            removed: Vec::new(),
            total_len: 0,
            active_docs: 0,
        }
    }

    pub fn tokenize(&self, text: &str) -> Vec<String> {
        tokenize(text, self.params.lowercase)
    }

    /// Index a document. `doc` ids must be appended in order (arena rows).
    pub fn add(&mut self, doc: u32, text: &str) {
        assert_eq!(doc as usize, self.doc_lengths.len(), "bm25 ids append-only");
        let tokens = self.tokenize(text);
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for t in &tokens {
            *counts.entry(t.as_str()).or_insert(0) += 1;
        }
        for (term, tf) in counts {
            self.postings
                .entry(term.to_string())
                .or_default()
                .push(Posting { doc, tf });
        }
        self.doc_lengths.push(tokens.len() as u32);
        self.removed.push(false);
        self.total_len += tokens.len() as u64;
        self.active_docs += 1;
    }

    /// Logically remove a document from statistics. Postings stay until
    /// compaction; the query-time `accept` predicate hides them.
    pub fn remove(&mut self, doc: u32) {
        if let Some(flag) = self.removed.get_mut(doc as usize) {
            if !*flag {
                *flag = true;
                self.total_len -= u64::from(self.doc_lengths[doc as usize]);
                self.active_docs -= 1;
            }
        }
    }

    pub fn active_docs(&self) -> u64 {
        self.active_docs
    }

    fn avg_len(&self) -> f32 {
        if self.active_docs == 0 {
            0.0
        } else {
            self.total_len as f32 / self.active_docs as f32
        }
    }

    /// BM25 top-k. `accept` integrates tombstones/filters; rejected docs are
    /// never scored. Returns (doc, score) best-first.
    pub fn search(
        &self,
        query: &str,
        k: usize,
        accept: &dyn Fn(u32) -> bool,
    ) -> Vec<(u32, f32)> {
        if k == 0 || self.active_docs == 0 {
            return Vec::new();
        }
        let terms = self.tokenize(query);
        if terms.is_empty() {
            return Vec::new();
        }
        let avg_len = self.avg_len();
        let n = self.active_docs as f32;
        let mut scores: HashMap<u32, f32> = HashMap::new();
        let mut seen_terms: std::collections::HashSet<&str> = std::collections::HashSet::new();
        for term in &terms {
            // Repeated query terms score once (standard practice).
            if !seen_terms.insert(term.as_str()) {
                continue;
            }
            let Some(postings) = self.postings.get(term.as_str()) else {
                continue;
            };
            // df over live docs only.
            let df = postings
                .iter()
                .filter(|p| !self.removed[p.doc as usize])
                .count() as f32;
            if df == 0.0 {
                continue;
            }
            let idf = ((n - df + 0.5) / (df + 0.5) + 1.0).ln();
            for p in postings {
                if self.removed[p.doc as usize] || !accept(p.doc) {
                    continue;
                }
                let tf = p.tf as f32;
                let dl = self.doc_lengths[p.doc as usize] as f32;
                let denom = tf
                    + self.params.k1 * (1.0 - self.params.b + self.params.b * dl / avg_len.max(1e-6));
                let score = idf * tf * (self.params.k1 + 1.0) / denom;
                *scores.entry(p.doc).or_insert(0.0) += score;
            }
        }
        let mut results: Vec<(u32, f32)> = scores.into_iter().collect();
        results.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.0.cmp(&b.0))
        });
        results.truncate(k);
        results
    }

    /// Number of distinct terms (for stats/doctor).
    pub fn term_count(&self) -> usize {
        self.postings.len()
    }
}

/// Unicode tokenizer: alphanumeric runs, optional lowercasing.
pub fn tokenize(text: &str, lowercase: bool) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut current = String::new();
    for ch in text.chars() {
        if ch.is_alphanumeric() {
            if lowercase {
                for lc in ch.to_lowercase() {
                    current.push(lc);
                }
            } else {
                current.push(ch);
            }
        } else if !current.is_empty() {
            tokens.push(std::mem::take(&mut current));
        }
    }
    if !current.is_empty() {
        tokens.push(current);
    }
    tokens
}

#[cfg(test)]
mod tests {
    use super::*;

    fn index_of(texts: &[&str]) -> Bm25Index {
        let mut idx = Bm25Index::new(Bm25Params::default());
        for (i, t) in texts.iter().enumerate() {
            idx.add(i as u32, t);
        }
        idx
    }

    #[test]
    fn tokenizer_handles_unicode_and_case() {
        assert_eq!(
            tokenize("Regras de Cancelamento, seção 2.1!", true),
            vec!["regras", "de", "cancelamento", "seção", "2", "1"]
        );
        assert_eq!(tokenize("", true), Vec::<String>::new());
        assert_eq!(tokenize("ÀÉÎ", true), vec!["àéî"]);
    }

    #[test]
    fn relevant_document_ranks_first() {
        let idx = index_of(&[
            "the quick brown fox jumps over the lazy dog",
            "cancellation policy: refunds within 30 days",
            "shipping information and delivery times",
        ]);
        let results = idx.search("cancellation refund policy", 3, &|_| true);
        assert_eq!(results[0].0, 1);
        assert!(results[0].1 > 0.0);
    }

    #[test]
    fn rare_terms_outweigh_common_terms() {
        let idx = index_of(&[
            "common common common word",
            "common word with zebra",
            "common word again",
        ]);
        let results = idx.search("zebra", 3, &|_| true);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 1);
    }

    #[test]
    fn removal_hides_documents_and_fixes_stats() {
        let mut idx = index_of(&["alpha beta", "alpha gamma", "alpha delta"]);
        assert_eq!(idx.active_docs(), 3);
        idx.remove(1);
        idx.remove(1); // idempotent
        assert_eq!(idx.active_docs(), 2);
        let results = idx.search("alpha", 10, &|_| true);
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|(d, _)| *d != 1));
    }

    #[test]
    fn accept_predicate_is_a_prefilter() {
        let idx = index_of(&["match here", "match there", "match everywhere"]);
        let results = idx.search("match", 10, &|d| d == 2);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 2);
    }

    #[test]
    fn empty_query_returns_empty() {
        let idx = index_of(&["something"]);
        assert!(idx.search("", 5, &|_| true).is_empty());
        assert!(idx.search("!!! ...", 5, &|_| true).is_empty());
    }

    #[test]
    fn serialization_round_trip() {
        let idx = index_of(&["alpha beta gamma", "beta gamma delta"]);
        let bytes = serde_json::to_vec(&idx).unwrap();
        let restored: Bm25Index = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(
            idx.search("beta gamma", 5, &|_| true),
            restored.search("beta gamma", 5, &|_| true)
        );
    }
}
