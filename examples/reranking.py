"""Plug a custom reranker (any callable; e.g. a cross-encoder)."""
import tempfile
from pathlib import Path

import ragvault

def keyword_boost_reranker(query, chunks):
    terms = set(query.lower().split())
    return sorted(
        chunks,
        key=lambda c: sum(t in c.text.lower() for t in terms),
        reverse=True,
    )

with ragvault.open(Path(tempfile.mkdtemp()) / "kb") as kb:
    kb.add(["alpha beta gamma", "alpha alpha alpha", "unrelated text"])
    result = kb.retrieve("alpha", k=2, rerank=keyword_boost_reranker)
    for c in result.chunks:
        print(c.score, c.text)
