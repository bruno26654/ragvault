"""RagVault — documents to a high-quality RAG knowledge base in one API.

    import ragvault

    kb = ragvault.open("./knowledge", preset="quality")
    kb.sync("./documents")
    result = kb.retrieve("Quais são as regras de cancelamento?", token_budget=6000)
    print(result.context)
    print(result.citations)
"""

from __future__ import annotations

from ._native import __version__ as _native_version
from .chunking import ChunkingConfig
from .config import Config, PRESETS
from .context import Citation, RetrievalResult, RetrievedChunk
from .embeddings import (
    Embedder,
    FunctionEmbedder,
    HashedNGramEmbedder,
    resolve_embedding,
)
from .errors import (
    ConfigurationError,
    DimensionMismatchError,
    EmbeddingError,
    EvaluationError,
    IngestionError,
    RagVaultError,
    VaultCorruptError,
    VaultError,
    VaultLockedError,
)
from .evaluate import EvaluationReport
from .kb import Answer, KnowledgeBase, SyncReport, open_kb as open  # noqa: A001
from .parsers import ParsedDocument, parse_file, register_parser

__version__ = _native_version

__all__ = [
    "open",
    "KnowledgeBase",
    "RetrievalResult",
    "RetrievedChunk",
    "Citation",
    "Answer",
    "SyncReport",
    "EvaluationReport",
    "Config",
    "PRESETS",
    "ChunkingConfig",
    "Embedder",
    "HashedNGramEmbedder",
    "FunctionEmbedder",
    "resolve_embedding",
    "ParsedDocument",
    "parse_file",
    "register_parser",
    "RagVaultError",
    "ConfigurationError",
    "EmbeddingError",
    "IngestionError",
    "EvaluationError",
    "VaultError",
    "VaultLockedError",
    "VaultCorruptError",
    "DimensionMismatchError",
    "__version__",
]
