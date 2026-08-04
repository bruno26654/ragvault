"""Structure-aware chunking.

Strategies: ``auto`` (default — picks per format), ``markdown`` (heading
hierarchy preserved as section paths), ``paragraph``, ``sentence``,
``recursive`` and ``fixed``. Chunks keep char offsets, section paths, token
counts and neighbor links so retrieval can expand context later.

Token counting is a pluggable callable; the default estimator
(``len(text) // 4`` refined by whitespace) is documented as an estimate.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable, Optional


#: Character ranges of scripts written without spaces between words.
#:
#: Kept in sync with `is_unspaced_script` in
#: `crates/ragvault-retrieval/src/bm25.rs`; `tests/python/test_components.py`
#: asserts the two agree, because a silent divergence between how text is
#: counted here and how it is tokenized there is exactly the class of bug this
#: constant exists to fix.
_UNSPACED_RANGES = (
    (0x3040, 0x309F),  # Hiragana
    (0x30A0, 0x30FF),  # Katakana
    (0x3400, 0x4DBF),  # CJK Unified Ideographs Extension A
    (0x4E00, 0x9FFF),  # CJK Unified Ideographs
    (0xF900, 0xFAFF),  # CJK Compatibility Ideographs
    (0xAC00, 0xD7AF),  # Hangul Syllables
    (0x0E00, 0x0E7F),  # Thai
    (0x0E80, 0x0EFF),  # Lao
    (0x1000, 0x109F),  # Myanmar
    (0x1780, 0x17FF),  # Khmer
)

#: Tokens per character for those scripts. Common BPE and SentencePiece
#: vocabularies emit roughly 0.6–1.0 tokens per Han character (more for rare
#: ones); this sits in the middle. It is an estimate and is documented as one —
#: `ChunkingConfig.tokenizer` takes a real tokenizer when the exact count
#: matters.
_UNSPACED_TOKENS_PER_CHAR = 0.7


def _is_unspaced_char(ch: str) -> bool:
    code = ord(ch)
    return any(lo <= code <= hi for lo, hi in _UNSPACED_RANGES)


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~chars/4 blended with word count).

    The word term only means anything in scripts that separate words with
    spaces. Counting a Chinese paragraph as "one word" made the estimate 5–9×
    low — 35 characters scored 4 tokens — so `target_tokens` and `token_budget`
    were both wrong by that factor and chunks silently overran whatever the
    caller budgeted. Characters in unspaced scripts are therefore counted on
    their own terms, and the original blend applies to the rest.
    """
    if not text:
        return 0
    unspaced = sum(1 for ch in text if _is_unspaced_char(ch))
    if not unspaced:
        words = len(text.split())
        return max(1, int(0.6 * words + 0.4 * (len(text) / 4)))
    spaced_text = "".join(ch for ch in text if not _is_unspaced_char(ch))
    words = len(spaced_text.split())
    spaced = 0.6 * words + 0.4 * (len(spaced_text) / 4)
    return max(1, int(unspaced * _UNSPACED_TOKENS_PER_CHAR + spaced))


@dataclass
class ChunkingConfig:
    strategy: str = "auto"
    target_tokens: int = 400
    max_tokens: int = 700
    overlap_tokens: int = 40
    tokenizer: Optional[Callable[[str], int]] = None

    def count(self, text: str) -> int:
        return (self.tokenizer or estimate_tokens)(text)

    def to_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "target_tokens": self.target_tokens,
            "max_tokens": self.max_tokens,
            "overlap_tokens": self.overlap_tokens,
            "tokenizer": "custom" if self.tokenizer else "estimate",
        }


@dataclass
class RawChunk:
    text: str
    chunk_index: int
    char_start: int
    char_end: int
    token_count: int
    section_path: list[str] = field(default_factory=list)
    page_number: Optional[int] = None


#: Sentence terminators of scripts without letter case, as Unicode
#: `Sentence_Terminal=Yes`. Lives here rather than in `verification` because
#: this module is the foundational one — it imports nothing internal — and both
#: splitters must agree on what ends a sentence.
CASELESS_TERMINATORS = (
    "。．！？"   # CJK / full-width stop, exclamation, question
    "｡"         # U+FF61 halfwidth ideographic full stop
    "।॥"        # U+0964/U+0965 danda, double danda — Indic
    "։"         # U+0589 Armenian full stop
    "።"         # U+1362 Ethiopic full stop
    "។"         # U+17D4 Khmer khan
    "၊။"        # U+104A/U+104B Myanmar little section, section
)

#: Sentence boundaries for chunking. Latin terminators need trailing
#: whitespace; caseless ones do not, because the scripts that use them do not
#: put spaces around punctuation.
#:
#: Only `[.!?…]` was recognized before, so a Chinese, Hindi or Arabic paragraph
#: was one sentence. That mattered because `_pack_units` falls back to sentence
#: splitting for an oversized paragraph — with no boundaries to find it went
#: straight to hard-wrapping, cutting mid-sentence in every language whose
#: terminator was missing.
_SENTENCE_RE = re.compile(
    rf"(?<=[.!?…])\s+|(?<=[{CASELESS_TERMINATORS}])\s*|(?<=[؟۔])\s+"
)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")


def _split_paragraphs(text: str) -> list[tuple[int, str]]:
    """Return (char_offset, paragraph) pairs."""
    parts: list[tuple[int, str]] = []
    offset = 0
    for block in re.split(r"\n\s*\n", text):
        if block.strip():
            start = text.index(block, offset)
            parts.append((start, block))
            offset = start + len(block)
    return parts


def _split_sentences(paragraph: str) -> list[str]:
    return [s for s in _SENTENCE_RE.split(paragraph) if s.strip()]


def _pack_units(
    units: list[tuple[int, str, list[str]]],
    cfg: ChunkingConfig,
) -> list[RawChunk]:
    """Greedy packing of (offset, text, section_path) units into chunks close
    to ``target_tokens``, hard-capped at ``max_tokens``, with sentence-level
    splitting of oversized units and token overlap between chunks."""
    chunks: list[RawChunk] = []
    buffer: list[tuple[int, str]] = []
    buffer_tokens = 0
    buffer_section: list[str] = []

    def flush() -> None:
        nonlocal buffer, buffer_tokens
        if not buffer:
            return
        text = "\n\n".join(part for _, part in buffer).strip()
        if text:
            start = buffer[0][0]
            end = buffer[-1][0] + len(buffer[-1][1])
            chunks.append(
                RawChunk(
                    text=text,
                    chunk_index=len(chunks),
                    char_start=start,
                    char_end=end,
                    token_count=cfg.count(text),
                    section_path=list(buffer_section),
                )
            )
        buffer = []
        buffer_tokens = 0

    for offset, unit, section in units:
        if section != buffer_section and buffer:
            flush()
        buffer_section = section
        unit_tokens = cfg.count(unit)
        if unit_tokens > cfg.max_tokens:
            flush()
            # Oversized paragraph: split by sentences, then hard-wrap.
            sentences = _split_sentences(unit) or [unit]
            piece: list[str] = []
            piece_tokens = 0
            piece_offset = offset
            for sentence in sentences:
                stoks = cfg.count(sentence)
                if piece and piece_tokens + stoks > cfg.target_tokens:
                    joined = " ".join(piece)
                    chunks.append(
                        RawChunk(
                            text=joined,
                            chunk_index=len(chunks),
                            char_start=piece_offset,
                            char_end=piece_offset + len(joined),
                            token_count=piece_tokens,
                            section_path=list(section),
                        )
                    )
                    # sentence-level overlap
                    keep = piece[-1] if cfg.overlap_tokens > 0 else None
                    piece = [keep] if keep else []
                    piece_tokens = cfg.count(keep) if keep else 0
                    piece_offset = offset
                while stoks > cfg.max_tokens:
                    # Pathological sentence: hard character wrap.
                    cut = max(1, len(sentence) * cfg.max_tokens // stoks)
                    head, sentence = sentence[:cut], sentence[cut:]
                    chunks.append(
                        RawChunk(
                            text=head,
                            chunk_index=len(chunks),
                            char_start=piece_offset,
                            char_end=piece_offset + len(head),
                            token_count=cfg.count(head),
                            section_path=list(section),
                        )
                    )
                    stoks = cfg.count(sentence)
                if sentence.strip():
                    piece.append(sentence)
                    piece_tokens += stoks
            if piece:
                joined = " ".join(piece)
                chunks.append(
                    RawChunk(
                        text=joined,
                        chunk_index=len(chunks),
                        char_start=piece_offset,
                        char_end=piece_offset + len(joined),
                        token_count=piece_tokens,
                        section_path=list(section),
                    )
                )
            continue
        if buffer and buffer_tokens + unit_tokens > cfg.target_tokens:
            flush()
            buffer_section = section
        buffer.append((offset, unit))
        buffer_tokens += unit_tokens
    flush()
    return chunks


def chunk_plain(text: str, cfg: ChunkingConfig) -> list[RawChunk]:
    units = [(off, para, []) for off, para in _split_paragraphs(text)]
    if not units and text.strip():
        units = [(0, text.strip(), [])]
    return _pack_units(units, cfg)  # type: ignore[arg-type]


def chunk_markdown(text: str, cfg: ChunkingConfig) -> list[RawChunk]:
    """Markdown chunking: heading hierarchy becomes section_path; headings
    open new chunks so sections never blur together."""
    units: list[tuple[int, str, list[str]]] = []
    section_stack: list[tuple[int, str]] = []
    offset = 0
    for line_block in re.split(r"\n\s*\n", text):
        if not line_block.strip():
            continue
        start = text.index(line_block, offset)
        offset = start + len(line_block)
        first_line = line_block.strip().splitlines()[0]
        match = _HEADING_RE.match(first_line)
        if match:
            level = len(match.group(1))
            title = match.group(2).strip()
            section_stack = [(lv, t) for lv, t in section_stack if lv < level]
            section_stack.append((level, title))
        units.append((start, line_block, [t for _, t in section_stack]))
    if not units and text.strip():
        units = [(0, text.strip(), [])]
    return _pack_units(units, cfg)


def chunk_text(text: str, cfg: ChunkingConfig, fmt: str = "text") -> list[RawChunk]:
    """Entry point: dispatch on strategy/format."""
    strategy = cfg.strategy
    if strategy == "auto":
        strategy = "markdown" if fmt in ("markdown", "md") else "recursive"
    if strategy in ("markdown",):
        return chunk_markdown(text, cfg)
    if strategy in ("recursive", "paragraph", "sentence", "auto"):
        return chunk_plain(text, cfg)
    if strategy == "fixed":
        fixed_cfg = ChunkingConfig(
            strategy="fixed",
            target_tokens=cfg.target_tokens,
            max_tokens=cfg.target_tokens,
            overlap_tokens=cfg.overlap_tokens,
            tokenizer=cfg.tokenizer,
        )
        return chunk_plain(text, fixed_cfg)
    raise ValueError(
        f"unknown chunking strategy {strategy!r}; expected one of "
        "auto/markdown/recursive/paragraph/sentence/fixed"
    )
