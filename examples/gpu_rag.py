"""GPU-accelerated dense retrieval via cuVS CAGRA (experimental).

Requires CUDA hardware and: pip install cuvs-cu12
Validation runbook: docs/GPU.md. On a machine without cuvs this exits with
an actionable message instead of pretending to work.
"""
import tempfile
from pathlib import Path

import ragvault
from ragvault.gpu import CagraDenseSearcher, is_gpu_available

if not is_gpu_available():
    raise SystemExit("cuvs not installed — pip install cuvs-cu12 (see docs/GPU.md)")

with ragvault.open(Path(tempfile.mkdtemp()) / "kb") as kb:
    kb.add([{"id": f"d{i}", "text": f"support article {i}"} for i in range(5000)])
    searcher = CagraDenseSearcher(kb)
    print(f"CAGRA build: {searcher.build_seconds:.2f}s over {len(searcher.chunk_ids)} vectors")
    result = kb.retrieve("support article 42", k=3, dense_searcher=searcher, explain=True)
    print("backend:", result.plan["dense_backend"])
    print("top hit:", result.chunks[0].document_id)
