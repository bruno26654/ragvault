"""Low-level multi-collection API.

The primary abstraction stays :class:`ragvault.KnowledgeBase`; ``Database``
is a thin organizer for applications that keep several knowledge bases
(collections) under one directory:

    db = ragvault.Database.open("./data")
    docs = db.collection("documents", preset="quality")
    faqs = db.collection("faqs")
    db.close()

Each collection is a full KnowledgeBase in ``<root>/<name>/`` with its own
configuration, WAL and snapshots.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Union

from .errors import ConfigurationError
from .kb import KnowledgeBase

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,63}$")


class Database:
    def __init__(self, root: Union[str, Path]) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self._collections: dict[str, KnowledgeBase] = {}

    @classmethod
    def open(cls, root: Union[str, Path]) -> "Database":
        return cls(root)

    def collection(self, name: str, **kwargs: object) -> KnowledgeBase:
        """Open (or create) a named collection. Names are restricted to a
        safe charset — no path traversal."""
        if not _NAME_RE.match(name):
            raise ConfigurationError(
                f"invalid collection name {name!r}: use letters, digits, "
                "'-', '_' or '.' (max 64 chars, must not start with a dot)"
            )
        if name not in self._collections:
            self._collections[name] = KnowledgeBase(self.root / name, **kwargs)  # type: ignore[arg-type]
        return self._collections[name]

    def list_collections(self) -> list[str]:
        found = set(self._collections)
        for child in self.root.iterdir():
            if child.is_dir() and (child / "ragvault.json").exists():
                found.add(child.name)
        return sorted(found)

    def close(self) -> None:
        for kb in self._collections.values():
            kb.close()
        self._collections.clear()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"Database(root={str(self.root)!r}, open_collections={sorted(self._collections)})"


def connect(url: str) -> KnowledgeBase:
    """Reserved for the future remote backend (same API as local KBs).

    Not implemented in v0.1 — local-first is the priority (ADR 0001). The
    signature exists so application code can already be written against
    one entry point.
    """
    raise NotImplementedError(
        "ragvault.connect() is planned but not implemented in v0.1: remote "
        "backends will speak the same KnowledgeBase API. Use "
        f"ragvault.open(path) for local vaults (got url={url!r})."
    )
