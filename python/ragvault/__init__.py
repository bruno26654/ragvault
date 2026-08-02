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
from .database import Database, connect
from .evaluate import EvaluationReport
from .kb import Answer, KnowledgeBase, SyncReport, open_kb as open  # noqa: A001
from .multiquery import MultiRetrievalResult
from .rerankers import maxsim_reranker, maxsim_score
from .verification import ClaimVerification, VerificationReport
from .parsers import ParsedDocument, parse_file, register_parser

# Prefer the installed distribution's version (always matches the wheel's
# pip metadata, e.g. "1.0.0rc1"); fall back to the native crate version for
# source/editable layouts where no distribution metadata exists.
try:  # pragma: no cover - depends on install layout
    from importlib.metadata import PackageNotFoundError, version as _dist_version

    try:
        __version__ = _dist_version("ragvault")
    except PackageNotFoundError:
        __version__ = _native_version
except ImportError:  # pragma: no cover - python < 3.8 never reaches here
    __version__ = _native_version

__all__ = [
    "open",
    "connect",
    "Database",
    "KnowledgeBase",
    "maxsim_reranker",
    "maxsim_score",
    "RetrievalResult",
    "MultiRetrievalResult",
    "VerificationReport",
    "ClaimVerification",
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
