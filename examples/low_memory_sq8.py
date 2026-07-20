"""SQ8 quantized backend: 4x smaller scans, no graph build, near-exact recall."""
import tempfile
from pathlib import Path

import ragvault

with ragvault.open(Path(tempfile.mkdtemp()) / "kb", preset="low_memory") as kb:
    kb.add([{"id": f"d{i}", "text": f"support article number {i}"} for i in range(200)])
    result = kb.retrieve("support article number 42", k=3, explain=True)
    print("backend:", result.plan["dense_backend"])   # sq8_flat
    print("top hit:", result.chunks[0].document_id)
    print("quantized bytes:", kb.stats()["sq8_bytes"])
