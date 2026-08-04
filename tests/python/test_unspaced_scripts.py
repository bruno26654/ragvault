"""Scripts that do not put spaces between words.

The library's foundations assumed whitespace was a word boundary. That is a
correct model for Latin, Cyrillic and Greek; for Chinese, Japanese, Korean and
Thai it is not a degraded model, it is the wrong one — and in every case it
failed silently, which is why these tests exist as reproductions rather than
as regression guards for a bug someone reported.
"""

from __future__ import annotations

import pytest

import ragvault
from ragvault import _native


ZH = "退款申请必须在购买后三十天内提交。"
JA = "返金の申請は購入後三十日以内に提出してください。"
TH = "คำขอคืนเงินต้องยื่นภายในสามสิบวันหลังจากการซื้อ"


class TestTokenizer:
    def test_spaced_scripts_are_unchanged(self):
        """The property that made the change safe to ship: no existing corpus
        shifts ranking."""
        assert _native.tokenize("Refund requests, filed!") == [
            "refund", "requests", "filed"]
        assert _native.tokenize("Возврат средств") == ["возврат", "средств"]

    def test_unspaced_scripts_become_bigrams(self):
        assert _native.tokenize("退款申请") == ["退款", "款申", "申请"]

    def test_scripts_split_at_the_boundary(self):
        assert _native.tokenize("退款refund申请") == ["退款", "refund", "申请"]

    def test_python_and_rust_agree_on_which_scripts_are_unspaced(self):
        """Two definitions of "unspaced" would drift, and the drift would be
        invisible: token counting would disagree with tokenization."""
        from ragvault.chunking import _is_unspaced_char

        for ch in "退款返金한국어คืนເງິນភាសាက":
            assert _is_unspaced_char(ch), f"{ch!r} unspaced in Rust, not Python"
            assert _native.tokenize(ch) == [ch]
        for ch in "abcБвгαβγمر":
            assert not _is_unspaced_char(ch), f"{ch!r} spaced in Rust, not Python"


class TestBm25Retrieval:
    """BM25 held one enormous term per chunk, so only a verbatim whole-string
    query could match and every substring query returned nothing."""

    @pytest.fixture
    def kb(self, tmp_path):
        base = ragvault.open(tmp_path / "kb", preset="offline-lite")
        base.add([
            {"id": "zh-refund", "text": ZH},
            {"id": "zh-ship", "text": "订单在五个工作日内发货到全国各地。"},
            {"id": "ja-refund", "text": JA},
            {"id": "th-refund", "text": TH},
            {"id": "en-refund", "text":
             "Refund requests must be filed within thirty days of purchase."},
        ])
        base.flush()
        yield base
        base.close()

    @pytest.mark.parametrize("query,expected", [
        ("退款", "zh-refund"),      # chinês: "refund"
        ("三十天", "zh-refund"),     # chinês: "thirty days"
        ("发货", "zh-ship"),         # chinês: "ship"
        ("返金", "ja-refund"),       # japonês: "refund"
        ("คืนเงิน", "th-refund"),   # tailandês: "refund"
    ])
    def test_a_substring_query_finds_the_document(self, kb, query, expected):
        hits = kb.retrieve(query, k=3, mode="keyword").chunks
        assert hits, f"{query!r} returned nothing"
        assert hits[0].document_id == expected

    def test_english_still_works(self, kb):
        hits = kb.retrieve("refund", k=3, mode="keyword").chunks
        assert hits and hits[0].document_id == "en-refund"

    def test_an_unrelated_query_does_not_match(self, kb):
        """Bigrams must not turn every CJK query into a match for everything."""
        assert not kb.retrieve("量子力学", k=3, mode="keyword").chunks


class TestDedupAndDiversity:
    """`_token_set` was `str.split()`, so two near-identical Chinese chunks
    scored 0.0 overlap and MMR admitted both as 'maximally diverse'."""

    def _overlap(self, a, b):
        from ragvault.context import _overlap, _token_set
        return _overlap(_token_set(a), _token_set(b))

    def test_near_duplicates_are_recognized(self):
        assert self._overlap(ZH, "退款申请必须在购买后三十天内向客服提交。") > 0.8
        assert self._overlap(TH, TH + "และการจัดส่ง") > 0.8

    def test_unrelated_chunks_are_not(self):
        assert self._overlap(ZH, "订单在五个工作日内发货到全国各地。") < 0.3

    def test_english_discrimination_is_not_lost(self):
        near = self._overlap(
            "Refund requests must be filed within thirty days.",
            "Refund requests must be filed within 30 days of purchase.")
        far = self._overlap(
            "Refund requests must be filed within thirty days.",
            "Orders ship within five business days.")
        assert near > far + 0.3


class TestTokenEstimation:
    """35 Chinese characters were estimated at 4 tokens, so `target_tokens`
    and `token_budget` were both wrong by 5-9x."""

    def test_unspaced_text_is_not_counted_as_one_word(self):
        from ragvault.chunking import estimate_tokens

        assert estimate_tokens("这是一个用来检查分词行为的测试句子") >= 10

    def test_spaced_text_is_unchanged(self):
        from ragvault.chunking import estimate_tokens

        text = "The quick brown fox jumps over the lazy dog every morning."
        assert estimate_tokens(text) == max(
            1, int(0.6 * len(text.split()) + 0.4 * (len(text) / 4)))

    def test_mixed_text_counts_both_populations(self):
        from ragvault.chunking import estimate_tokens

        assert (estimate_tokens("The rule 退款申请必须提交 applies.")
                > estimate_tokens("The rule applies."))

    def test_chunks_respect_the_budget_in_cjk(self, tmp_path):
        """The end the estimate serves: a Chinese document must not produce
        chunks far past max_tokens."""
        from ragvault.chunking import ChunkingConfig, chunk_text, estimate_tokens

        cfg = ChunkingConfig(target_tokens=40, max_tokens=70, overlap_tokens=0)
        text = "。".join("退款申请必须在购买后三十天内提交" for _ in range(40))
        for c in chunk_text(text, cfg):
            assert estimate_tokens(c.text) <= cfg.max_tokens * 1.5


class TestChunkSentenceSplitting:
    """`_pack_units` falls back to sentence splitting for an oversized
    paragraph. With no terminators recognized it went straight to hard-wrap,
    cutting mid-sentence in every script whose terminator was missing."""

    @pytest.mark.parametrize("text", [
        "第一句话在这里。第二句话在这里。第三句。",
        "पहला वाक्य यहाँ है। दूसरा वाक्य यहाँ है। तीसरा।",
        "الجملة الأولى هنا؟ الجملة الثانية هنا؟ الثالثة.",
        "ይህ የመጀመሪያው ነው። ይህ ሁለተኛው ነው። ሦስተኛው።",
        "First sentence here. Second sentence here. Third one.",
    ])
    def test_three_sentences_are_found(self, text):
        from ragvault.chunking import _split_sentences

        assert len(_split_sentences(text)) == 3

    def test_the_chunker_and_the_verifier_share_one_definition(self):
        from ragvault.chunking import CASELESS_TERMINATORS
        from ragvault.verification import _CASELESS_TERMINATORS

        assert _CASELESS_TERMINATORS is CASELESS_TERMINATORS
