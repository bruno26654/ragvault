"""Faiss interop (convertible level): export vectors to faiss and back.

Requires: pip install "ragvault[faiss]"
"""
import tempfile
from pathlib import Path

import ragvault
from ragvault.compat import faiss as rv_faiss

with ragvault.open(Path(tempfile.mkdtemp()) / "kb") as kb:
    kb.add([{"id": "a", "text": "cancellation policy"},
            {"id": "b", "text": "shipping schedule"}])
    index, chunk_ids = rv_faiss.export_to_faiss(kb)
    print("exported to faiss:", index.ntotal, "vectors;", chunk_ids)
    vectors = rv_faiss.reconstruct_from_faiss(index, chunk_ids)
    print("reconstructed shape:", vectors.shape)
