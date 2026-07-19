"""Native retrieval evaluation with a JSONL dataset."""
import json
import tempfile
from pathlib import Path

import ragvault

base = Path(tempfile.mkdtemp())
with ragvault.open(base / "kb") as kb:
    kb.add([
        {"id": "refunds", "text": "Refunds are processed within 30 days."},
        {"id": "shipping", "text": "Orders ship in five business days."},
    ])
    dataset = base / "eval.jsonl"
    dataset.write_text(
        json.dumps({"query": "refund timing", "relevant_ids": ["refunds"]}) + "\n"
        + json.dumps({"query": "how fast is shipping", "relevant_ids": ["shipping"]}) + "\n"
    )
    report = kb.evaluate(dataset, k=2)
    print(report.to_markdown())
