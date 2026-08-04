# ADR 0017 — Tokenization for scripts without word spacing

Status: **implemented and validated**.
Introduces `format_version = 3` (no on-disk layout change; it marks the BM25
index as needing a rebuild).

## Context

The BM25 tokenizer accumulated runs of `char::is_alphanumeric`. Every CJK
character is alphanumeric and nothing separates them, so **an entire Chinese,
Japanese, Korean or Thai chunk became one token**. Reproduced before the change:

```
en: "refund"                      -> [('en-refund', 0.77)]     ok
en: "thirty days"                 -> [('en-refund', 1.55)]     ok
zh: "退款"        (refund)         -> NO RESULTS
zh: "三十天"      (thirty days)    -> NO RESULTS
ja: "返金"        (refund)         -> NO RESULTS
th: "คืนเงิน"     (refund)         -> NO RESULTS
zh: <entire chunk, verbatim>      -> [('zh-refund', 1.95)]     only exact match
```

`hybrid` is the default retrieval mode in every preset, including
`multilingual`. For these corpora the BM25 half contributed nothing, so hybrid
degraded to dense-only — silently, and against an advertised capability.

The same assumption appeared in two more places, from the same root cause
("whitespace is a word boundary"):

- `context.py::_token_set` used `str.split()`, so MMR's overlap between two
  near-identical Chinese chunks was **0.00**. MMR reads that as "maximally
  diverse" and admitted both, spending context budget on a duplicate — the
  exact outcome MMR exists to prevent.
- `chunking.py::estimate_tokens` blended a word count, and 35 Chinese
  characters estimated at **4 tokens** (5–9× low). `target_tokens` and
  `token_budget` were both wrong by that factor.

## Decision

**Overlapping character bigrams for scripts written without word spacing**, the
behaviour of Lucene's `CJKBigramFilter` and Elasticsearch's `cjk` analyzer.

Chosen for the reason it is chosen there: it needs no dictionary, no model and
no per-language data, so it degrades predictably instead of requiring a resource
the core cannot carry. Ranges covered: Hiragana, Katakana, CJK Unified (+ Ext A,
Compatibility), Hangul, Thai, Lao, Myanmar, Khmer.

- Spaced scripts are untouched — Latin, Cyrillic, Greek and Arabic tokenize
  byte-identically to before, which is the property that made this shippable
  without re-ranking existing corpora. Asserted directly in
  `bm25::tokenizer_tests::spaced_scripts_are_unchanged`.
- Hangul is included although Korean *is* spaced: Korean is agglutinative, so
  particles attach to the stem and bigrams inside each spaced word are what let
  those forms match. Word boundaries still cut the run.
- A run of one character emits that character. Runs of two or more emit bigrams
  only, so a one-character query does not match bigrammed text — the documented
  Lucene trade-off, taken rather than inflating every posting list with
  unigrams.

**One tokenizer, not two.** `tokenize` is re-exported through the engine and
bound into `ragvault._native`, and `_token_set` calls it. A second definition in
Python would drift, and a drift between how the index tokenizes and how dedup
tokenizes is invisible until it misbehaves.

**Token estimation counts unspaced characters on their own terms** at ~0.7
tokens/char, documented as an estimate; `ChunkingConfig.tokenizer` already takes
a real tokenizer when the exact count matters. The Python range table is
asserted against the Rust one by test, since these are the two definitions that
cannot be shared.

## Migration

The BM25 index is persisted, so postings written by an older build hold terms
the current tokenizer can never produce. `format_version = 3` marks this;
opening a vault below v3 rebuilds BM25 from the stored chunk text, reusing what
compaction already does, and the next flush records v3. Nothing else on disk
changes.

The rebuild is O(corpus) on first open and a no-op in effect for Latin corpora,
which tokenize identically before and after — they pay the cost and get the same
index. Deleted rows and holes in the row space are preserved
(`pre_tokenizer_vault_rebuilds_bm25_on_reopen`).

## Consequences

- Substring queries work in CJK and Thai; `hybrid` is genuinely hybrid there.
- Index size grows for these scripts: a run of *n* characters yields *n−1*
  bigrams rather than 1 token. This is the standard cost of the approach.
- Thai, Lao, Khmer and Burmese are **usable, not solved**. Bigrams take them
  from one token per chunk to matchable; dictionary segmentation would do
  better and is deliberately not in the core, the same trade-off the
  verification layer resolves with a caller-supplied `segmenter=`.
- A single-character CJK query does not match. Documented.

## Alternatives rejected

- **Dictionary segmentation (jieba, ICU, Lindera)** — better quality, but puts
  per-language data and a dependency in a core that has to stay small, for the
  same reason ICU4X was declined for sentence segmentation.
- **Unigrams as well as bigrams** — fixes the single-character query, roughly
  doubles posting lists and dilutes IDF for every term.
- **Leaving it and documenting the limitation** — the `multilingual` preset
  advertises the opposite, and the failure was silent, which is what makes it a
  bug rather than a limit.
