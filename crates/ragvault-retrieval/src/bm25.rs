//! Incremental BM25 inverted index.
//!
//! - Unicode-aware tokenizer: lowercased alphanumeric runs for spaced scripts,
//!   overlapping character bigrams for scripts written without word spacing.
//!   The bigrams are not an optimization — without them an alphanumeric run
//!   *is* the whole sentence in Chinese, Japanese, Korean and Thai, so the
//!   index held one enormous term per chunk and only a verbatim whole-string
//!   query could match. Every substring query returned nothing, and because
//!   `hybrid` is the default mode everywhere it degraded to dense-only in
//!   silence. (This comment previously claimed such text "degrades to
//!   character runs"; it did not.)
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
    pub fn search(&self, query: &str, k: usize, accept: &dyn Fn(u32) -> bool) -> Vec<(u32, f32)> {
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
                    + self.params.k1
                        * (1.0 - self.params.b + self.params.b * dl / avg_len.max(1e-6));
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

/// Scripts written without spaces between words, where an alphanumeric run is
/// a sentence rather than a word.
///
/// Hangul is included even though Korean *is* spaced: Korean is agglutinative,
/// so particles attach to the stem ("한국어를" / "한국어는") and bigrams inside
/// each spaced word are what let those forms match each other. Word boundaries
/// still cut the run, so this only ever bigrams within one word.
fn is_unspaced_script(ch: char) -> bool {
    matches!(ch as u32,
        0x3040..=0x309F   // Hiragana
        | 0x30A0..=0x30FF // Katakana
        | 0x3400..=0x4DBF // CJK Unified Ideographs Extension A
        | 0x4E00..=0x9FFF // CJK Unified Ideographs
        | 0xF900..=0xFAFF // CJK Compatibility Ideographs
        | 0xAC00..=0xD7AF // Hangul Syllables
        | 0x0E00..=0x0E7F // Thai
        | 0x0E80..=0x0EFF // Lao
        | 0x1000..=0x109F // Myanmar
        | 0x1780..=0x17FF // Khmer
    )
}

fn flush_spaced(buf: &mut String, out: &mut Vec<String>) {
    if !buf.is_empty() {
        out.push(std::mem::take(buf));
    }
}

/// Overlapping character bigrams, the Lucene `CJKBigramFilter` behaviour.
///
/// A dictionary would segment these scripts better, but it would put
/// per-language data in a core that has to stay small — the same trade-off the
/// verification layer resolves with a caller-supplied `segmenter`. Bigrams need
/// nothing and take these languages from "one token per chunk" to usable.
///
/// A run of one character emits that character, so a single-glyph term is not
/// lost. Runs of two or more emit bigrams only: a one-character *query* against
/// bigrammed text therefore does not match, which is the documented Lucene
/// trade-off and the price of not inflating every posting list with unigrams.
fn flush_unspaced(buf: &mut Vec<char>, out: &mut Vec<String>) {
    if buf.is_empty() {
        return;
    }
    if buf.len() == 1 {
        out.push(buf[0].to_string());
    } else {
        for pair in buf.windows(2) {
            out.push(pair.iter().collect());
        }
    }
    buf.clear();
}

/// Unicode tokenizer: alphanumeric runs for spaced scripts, overlapping
/// character bigrams for unspaced ones, optional lowercasing.
///
/// Used by both indexing and querying, which is what makes the two consistent
/// by construction — a tokenizer change can never desynchronize them.
pub fn tokenize(text: &str, lowercase: bool) -> Vec<String> {
    let mut tokens = Vec::new();
    let mut spaced = String::new();
    let mut unspaced: Vec<char> = Vec::new();

    for ch in text.chars() {
        if is_unspaced_script(ch) {
            // Checked before `is_alphanumeric`, which is true for these too.
            flush_spaced(&mut spaced, &mut tokens);
            // No cased characters exist in these ranges, so lowercasing is a
            // no-op and the char is kept as-is.
            unspaced.push(ch);
        } else if ch.is_alphanumeric() {
            flush_unspaced(&mut unspaced, &mut tokens);
            if lowercase {
                for lc in ch.to_lowercase() {
                    spaced.push(lc);
                }
            } else {
                spaced.push(ch);
            }
        } else {
            flush_spaced(&mut spaced, &mut tokens);
            flush_unspaced(&mut unspaced, &mut tokens);
        }
    }
    flush_spaced(&mut spaced, &mut tokens);
    flush_unspaced(&mut unspaced, &mut tokens);
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

#[cfg(test)]
mod tokenizer_tests {
    use super::tokenize;

    /// The property that makes this change safe to ship: spaced scripts must
    /// tokenize exactly as before, so no existing corpus shifts ranking.
    #[test]
    fn spaced_scripts_are_unchanged() {
        assert_eq!(
            tokenize("Refund requests, filed!", true),
            vec!["refund", "requests", "filed"]
        );
        assert_eq!(
            tokenize("Возврат средств", true),
            vec!["возврат", "средств"]
        );
        assert_eq!(tokenize("استرداد الأموال", true), vec!["استرداد", "الأموال"]);
        assert_eq!(tokenize("R$ 1.234,56", true), vec!["r", "1", "234", "56"]);
    }

    #[test]
    fn cjk_becomes_overlapping_bigrams() {
        // Was one token: the whole run, matchable only by a verbatim query.
        assert_eq!(tokenize("退款申请", true), vec!["退款", "款申", "申请"]);
    }

    #[test]
    fn a_substring_query_shares_a_term_with_the_document() {
        let doc = tokenize("退款申请必须提交", true);
        let query = tokenize("退款", true);
        assert!(
            query.iter().any(|t| doc.contains(t)),
            "no shared term: doc={doc:?} query={query:?}"
        );
    }

    #[test]
    fn scripts_split_at_the_boundary() {
        assert_eq!(
            tokenize("退款refund申请", true),
            vec!["退款", "refund", "申请"]
        );
        assert_eq!(tokenize("30天", true), vec!["30", "天"]);
    }

    #[test]
    fn a_single_character_run_survives() {
        assert_eq!(tokenize("退 款", true), vec!["退", "款"]);
    }

    #[test]
    fn japanese_korean_and_thai_are_covered() {
        assert_eq!(
            tokenize("返金の申請", true),
            vec!["返金", "金の", "の申", "申請"]
        );
        // Korean is spaced, so bigrams stay inside each word.
        assert_eq!(
            tokenize("한국어를 배우다", true),
            vec!["한국", "국어", "어를", "배우", "우다"]
        );
        assert!(tokenize("คืนเงิน", true).len() > 1);
    }

    #[test]
    fn lowercasing_still_applies_to_cased_scripts_only() {
        assert_eq!(tokenize("Refund 退款", false), vec!["Refund", "退款"]);
        assert_eq!(tokenize("Refund 退款", true), vec!["refund", "退款"]);
    }
}
