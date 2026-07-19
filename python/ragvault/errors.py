"""RagVault exception hierarchy.

Native-engine errors (lock conflicts, corruption, dimension mismatches) are
raised as the exception classes defined in the compiled module; this module
re-exports them and adds the Python-layer errors so users can catch
``ragvault.errors.RagVaultError`` for everything.
"""

from __future__ import annotations

from ragvault._native import (  # noqa: F401  (re-exports)
    DimensionMismatchError,
    VaultCorruptError,
    VaultError,
    VaultLockedError,
)


class RagVaultError(Exception):
    """Base class for Python-layer RagVault errors."""


class ConfigurationError(RagVaultError):
    """Invalid or incompatible configuration (preset, embedding, chunking)."""


class EmbeddingError(RagVaultError):
    """An embedder failed or returned malformed output."""


class IngestionError(RagVaultError):
    """A document could not be parsed or ingested.

    Carries file, parser, stage and a suggested fix when available.
    """

    def __init__(self, message: str, *, file: str | None = None,
                 parser: str | None = None, stage: str | None = None,
                 suggestion: str | None = None) -> None:
        details = []
        if file:
            details.append(f"file={file}")
        if parser:
            details.append(f"parser={parser}")
        if stage:
            details.append(f"stage={stage}")
        full = message if not details else f"{message} ({', '.join(details)})"
        if suggestion:
            full = f"{full}. Suggestion: {suggestion}"
        super().__init__(full)
        self.file = file
        self.parser = parser
        self.stage = stage
        self.suggestion = suggestion


class EvaluationError(RagVaultError):
    """Evaluation dataset missing or malformed."""
