"""Multi-tenant isolation: writes tagged, queries scoped automatically."""
import tempfile
from pathlib import Path

import ragvault

with ragvault.open(Path(tempfile.mkdtemp()) / "kb") as kb:
    kb.for_tenant("acme").add([{"id": "a", "text": "acme quarterly plan"}])
    kb.for_tenant("globex").add([{"id": "g", "text": "globex quarterly plan"}])
    print([c.document_id for c in kb.for_tenant("acme").retrieve("quarterly plan").chunks])
    # -> only acme documents; filter injection cannot escape the tenant scope
