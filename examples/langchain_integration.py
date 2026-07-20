"""LangChain integration (requires: pip install langchain-core)."""
import tempfile
from pathlib import Path

import ragvault

with ragvault.open(Path(tempfile.mkdtemp()) / "kb") as kb:
    kb.add([{"id": "refunds", "text": "Refunds are processed within 30 days."}])
    retriever = kb.as_langchain_retriever(k=2)
    docs = retriever.invoke("refund timing")
    for doc in docs:
        print(doc.metadata["document_id"], "->", doc.page_content)
