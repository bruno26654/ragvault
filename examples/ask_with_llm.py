"""kb.ask with a user-provided LLM callable (provider-neutral)."""
import tempfile
from pathlib import Path

import ragvault

def toy_llm(prompt: str) -> str:
    # Replace with a call to your provider of choice.
    return "According to [1], refunds are processed within 30 days."

with ragvault.open(Path(tempfile.mkdtemp()) / "kb") as kb:
    kb.add([{"id": "refunds", "text": "Refunds are processed within 30 days."}])
    answer = kb.ask("how long do refunds take?", llm=toy_llm)
    print(answer.text)
    print(answer.citations)
