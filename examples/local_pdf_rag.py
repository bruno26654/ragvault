"""PDF RAG (requires: pip install "ragvault[pdf]"). Points at your own PDFs."""
import sys
import tempfile
from pathlib import Path

import ragvault

if len(sys.argv) != 2:
    raise SystemExit("usage: python local_pdf_rag.py <directory-with-pdfs>")

with ragvault.open(Path(tempfile.mkdtemp()) / "kb", preset="quality",
                   embedding="builtin:hashed-ngram") as kb:
    report = kb.sync(sys.argv[1], include=["**/*.pdf"])
    print(report)
    result = kb.retrieve("summarize the main obligations", token_budget=4000)
    print(result.context[:2000])
