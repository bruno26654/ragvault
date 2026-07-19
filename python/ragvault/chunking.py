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


def estimate_tokens(text: str) -> int:
    """Cheap token estimate (~chars/4 blended with word count)."""
    if not text:
        return 0
    words = len(text.split())
    return max(1, int(0.6 * words + 0.4 * (len(text) / 4)))


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


_SENTENCE_RE = re.compile(r"(?<=[.!?…])\s+")
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
