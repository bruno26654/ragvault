"""Official framework adapters (optional dependencies, explicit imports).

Each adapter fails with an actionable message when its framework is not
installed. RagVault itself never imports these frameworks at package import
time.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .errors import ConfigurationError

if TYPE_CHECKING:  # pragma: no cover
    from .kb import KnowledgeBase


def as_langchain_retriever(kb: "KnowledgeBase", *, k: int = 8, **retrieve_kwargs: Any):
    """A LangChain ``BaseRetriever`` backed by this knowledge base.

    Requires ``pip install langchain-core``. Retrieved chunks map to
    LangChain Documents with full provenance metadata.
    """
    try:
        from langchain_core.callbacks import CallbackManagerForRetrieverRun
        from langchain_core.documents import Document as LCDocument
        from langchain_core.retrievers import BaseRetriever
    except ImportError as exc:
        raise ConfigurationError(
            "as_langchain_retriever() requires langchain-core: "
            "pip install langchain-core"
        ) from exc

    class RagVaultRetriever(BaseRetriever):
        """LangChain retriever over a RagVault KnowledgeBase."""

        _kb: Any = None
        _k: int = 8
        _kwargs: dict = {}

        def _get_relevant_documents(
            self, query: str, *, run_manager: "CallbackManagerForRetrieverRun"
        ) -> list["LCDocument"]:
            result = self._kb.retrieve(query, k=self._k, **self._kwargs)
            docs = []
            for chunk in result.chunks:
                docs.append(
                    LCDocument(
                        page_content=chunk.text,
                        metadata={
                            "document_id": chunk.document_id,
                            "document_version": chunk.document_version,
                            "chunk_id": chunk.chunk_id,
                            "chunk_index": chunk.chunk_index,
                            "title": chunk.title,
                            "uri": chunk.uri,
                            "section_path": chunk.section_path,
                            "page_number": chunk.page_number,
                            "score": chunk.score,
                            "expanded": chunk.expanded,
                            **chunk.metadata,
                        },
                    )
                )
            return docs

    retriever = RagVaultRetriever()
    retriever._kb = kb
    retriever._k = k
    retriever._kwargs = dict(retrieve_kwargs)
    return retriever


def as_llamaindex_retriever(kb: "KnowledgeBase", *, k: int = 8, **retrieve_kwargs: Any):
    """A LlamaIndex ``BaseRetriever`` backed by this knowledge base.

    Requires ``pip install llama-index-core``.
    """
    try:
        from llama_index.core.retrievers import BaseRetriever
        from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
    except ImportError as exc:
        raise ConfigurationError(
            "as_llamaindex_retriever() requires llama-index-core: "
            "pip install llama-index-core"
        ) from exc

    class RagVaultLlamaRetriever(BaseRetriever):
        def _retrieve(self, query_bundle: "QueryBundle") -> list["NodeWithScore"]:
            result = kb.retrieve(query_bundle.query_str, k=k, **retrieve_kwargs)
            nodes = []
            for chunk in result.chunks:
                node = TextNode(
                    id_=chunk.chunk_id,
                    text=chunk.text,
                    metadata={
                        "document_id": chunk.document_id,
                        "document_version": chunk.document_version,
                        "title": chunk.title,
                        "uri": chunk.uri,
                        "section_path": chunk.section_path,
                    },
                )
                nodes.append(NodeWithScore(node=node, score=chunk.score))
            return nodes

    return RagVaultLlamaRetriever()
