"""Quickstart: documents to cited context, fully offline."""
import tempfile
from pathlib import Path

import ragvault

docs = Path(tempfile.mkdtemp()) / "documents"
docs.mkdir()
(docs / "cancellation.md").write_text(
    "# Cancellation Policy\n\nCustomers may cancel within 30 days for a full refund.\n"
)
(docs / "shipping.md").write_text("# Shipping\n\nOrders ship within 5 business days.\n")

with ragvault.open(Path(tempfile.mkdtemp()) / "kb", preset="quality") as kb:
    print(kb.sync(docs))
    result = kb.retrieve("how do refunds work?", token_budget=2000)
    print(result.context)
    for c in result.citations:
        print(f"[{c.index}] {c.title} (doc={c.document_id}, v{c.document_version})")
