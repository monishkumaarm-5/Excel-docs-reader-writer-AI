from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from src.doxl_ai_terminal.data_structure.excel import Chunk


def get_embedding_model():
    """Single shared local embedding model."""

    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )


def store_in_chroma(
    chunks: List[Chunk],
    collection_name: str
) -> tuple:
    """
    Convert Chunks → LangChain Documents → Chroma vectorstore.

    Returns:
        (vectorstore, retriever)
    """

    documents = []

    for chunk in chunks:
        doc = Document(
            page_content=chunk.text,
            metadata={
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
            }
        )

        documents.append(doc)

    # Local embedding model
    embeddings = get_embedding_model()

    # Store embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory="./vector_store",
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return vectorstore, retriever