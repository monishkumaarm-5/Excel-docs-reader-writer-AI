#vector_db_operation.py
"""Vector database CRUD operations."""

from __future__ import annotations

from typing import List, Optional, Tuple

from langchain_chroma import Chroma
from langchain_core.documents import Document

from doxl_ai_terminal.config import VECTOR_STORE_DIR
from doxl_ai_terminal.data_handler.vector_config import get_embedding_model
from doxl_ai_terminal.data_structure.excel import Chunk


class VectorDBManager:
    """Full CRUD manager for the Chroma vector store.

    Usage::

        mgr = VectorDBManager()
        mgr.load_collection("my_collection")
        results = mgr.search("some query", k=5)
    """

    def __init__(self, persist_directory: Optional[str] = None):
        self.persist_directory = persist_directory or VECTOR_STORE_DIR
        self._embeddings = get_embedding_model()
        self._vectorstore: Optional[Chroma] = None
        self._collection_name: Optional[str] = None

    # ── Load ──

    def load_collection(self, collection_name: str) -> "VectorDBManager":
        """Load an existing persisted Chroma collection."""
        self._collection_name = collection_name
        self._vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        return self

    # ── Search ──

    def search(self, query: str, k: int = 4) -> List[Document]:
        """Similarity search, returns top-k documents."""
        self._assert_loaded()
        return self._vectorstore.similarity_search(query, k=k)

    def search_with_scores(
        self, query: str, k: int = 4
    ) -> List[Tuple[Document, float]]:
        """Similarity search with relevance scores."""
        self._assert_loaded()
        return self._vectorstore.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int = 4):
        """Return a LangChain retriever for the loaded collection."""
        self._assert_loaded()
        return self._vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    # ── Add ──

    def add_chunks(self, chunks: List[Chunk]) -> int:
        """Add new chunks to the loaded collection. Returns count added."""
        self._assert_loaded()
        documents = [
            Document(
                page_content=chunk.text,
                metadata={**chunk.metadata, "chunk_id": chunk.chunk_id},
            )
            for chunk in chunks
        ]
        self._vectorstore.add_documents(documents)
        return len(documents)

    def add_documents(self, documents: List[Document]) -> int:
        """Add raw LangChain Documents to the loaded collection."""
        self._assert_loaded()
        self._vectorstore.add_documents(documents)
        return len(documents)

    # ── Delete ──

    def delete_collection(self, collection_name: Optional[str] = None) -> str:
        """Delete a collection from the vector store."""
        name = collection_name or self._collection_name
        if not name:
            return "No collection specified."

        store = Chroma(
            collection_name=name,
            embedding_function=self._embeddings,
            persist_directory=self.persist_directory,
        )
        store.delete_collection()
        if name == self._collection_name:
            self._vectorstore = None
            self._collection_name = None
        return f"Deleted collection: {name}"

    # ── List ──

    def list_collections(self) -> List[str]:
        """List all collection names in the persist directory."""
        import chromadb

        client = chromadb.PersistentClient(path=self.persist_directory)
        return [col.name for col in client.list_collections()]

    # ── Info ──

    def collection_count(self) -> int:
        """Return the number of documents in the loaded collection."""
        self._assert_loaded()
        return self._vectorstore._collection.count()

    @property
    def collection_name(self) -> Optional[str]:
        return self._collection_name

    # ── Internal ──

    def _assert_loaded(self):
        if self._vectorstore is None:
            raise RuntimeError(
                "No collection loaded. Call load_collection() first."
            )
