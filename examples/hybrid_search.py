"""Hybrid retrieval: dense + BM25 fusion, filters and explain."""
import tempfile
from pathlib import Path

import ragvault

with ragvault.open(Path(tempfile.mkdtemp()) / "kb") as kb:
    kb.add([
        {"id": "err-1", "text": "Error XK-4211 means the payment gateway timed out.",
         "metadata": {"kind": "errors"}},
        {"id": "faq-1", "text": "Payments are processed by our gateway partners.",
         "metadata": {"kind": "faq"}},
    ])
    result = kb.retrieve("what does XK-4211 mean?", k=2, explain=True)
    print(result.context)
    print("plan:", result.plan)

    only_faq = kb.retrieve("payments", filters={"kind": "faq"}, k=5)
    print("filtered:", [c.document_id for c in only_faq.chunks])
