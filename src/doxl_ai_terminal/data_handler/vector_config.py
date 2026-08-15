#vector_config.py
import hashlib
import os
import re
from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from doxl_ai_terminal.data_structure.excel import Chunk
from doxl_ai_terminal.config import VECTOR_STORE_DIR

# ── Local / offline embedding config ──────────────────────────────
# Both the vector DB (Chroma, above) and the embedding model run
# entirely on-device — no API keys, no network calls at inference
# time. The embedding model's weights are cached inside the project
# (EMBEDDING_CACHE_DIR) instead of the user's global ~/.cache, so the
# whole thing is self-contained and portable.
EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_CACHE_DIR = str(Path(VECTOR_STORE_DIR).parent / "models" / "embeddings")
os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)


# Cached singleton so repeated get_embedding_model() calls (e.g. one
# per write, when re-indexing after every edit) don't reload the
# sentence-transformers weights from disk each time.
_EMBEDDING_MODEL_SINGLETON = None


def get_embedding_model(offline: bool = True):
    """Single shared local embedding model.

    Runs fully on-device via sentence-transformers — never calls out
    to any API for embeddings.

    When `offline=True` (the default), this also forces the Hugging
    Face Hub client into offline mode (HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE),
    so it will NOT attempt any network request — it only uses whatever
    is already cached in EMBEDDING_CACHE_DIR and raises immediately if
    the model isn't there yet, instead of hanging on a network call.

    First-time setup (run once, with internet access, to populate the
    cache):
        python -m doxl_ai_terminal.data_handler.download_embedding_model

    After that, every call to get_embedding_model() works with zero
    network access.
    """
    global _EMBEDDING_MODEL_SINGLETON
    if _EMBEDDING_MODEL_SINGLETON is not None:
        return _EMBEDDING_MODEL_SINGLETON

    if offline:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    _EMBEDDING_MODEL_SINGLETON = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_DIR,
        model_kwargs={
            "device": "cpu"
        },
        encode_kwargs={
            "normalize_embeddings": True
        }
    )
    return _EMBEDDING_MODEL_SINGLETON


def make_collection_name(filepath: str, format_name: str) -> str:
    """
    Build a Chroma collection name that's unique per actual FILE
    LOCATION, not just per filename.

    Bug this fixes: the old scheme was `f"{basename}_{format_name}"`
    (e.g. "report_sub_para"). Two different files both named
    "report.docx" in two different folders would collide on that same
    collection name and silently share/overwrite each other's
    embeddings — vector_search could return chunks from the wrong
    file entirely. Hashing the absolute, case-normalized path makes
    every real file location get its own collection, while keeping
    the plain filename as a readable prefix.
    """
    abs_path = os.path.abspath(filepath)
    # normcase so the same file always hashes the same way regardless
    # of how the path was capitalized/spelled (Windows paths are
    # case-insensitive; this keeps things stable across runs).
    norm_path = os.path.normcase(abs_path)
    path_hash = hashlib.sha1(norm_path.encode("utf-8")).hexdigest()[:10]

    filename = os.path.splitext(os.path.basename(filepath))[0]
    # Chroma collection names must be 3-63 chars, and may only contain
    # [a-zA-Z0-9_-], starting/ending with an alphanumeric. Sanitize the
    # human-readable part so odd filenames can't produce an invalid name.
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", filename).strip("_-")[:40]
    safe_format = re.sub(r"[^a-zA-Z0-9_-]", "_", format_name)

    return f"{safe_name or 'file'}_{path_hash}_{safe_format}"


def store_in_chroma(
    chunks: List[Chunk],
    collection_name: str
) -> tuple:
    """
    Convert Chunks → LangChain Documents → Chroma vectorstore.

    Documents are stored with explicit ids = chunk.chunk_id, so
    re-embedding the same chunk (e.g. re-indexing after a write, or
    reopening the same file in a new session) overwrites the existing
    vector for that chunk instead of silently piling up a duplicate
    next to it.

    Returns:
        (vectorstore, retriever)
    """

    documents = []
    ids = []

    for chunk in chunks:
        doc = Document(
            page_content=chunk.text,
            metadata={
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
            }
        )

        documents.append(doc)
        ids.append(chunk.chunk_id)

    # Local embedding model
    embeddings = get_embedding_model()

    # Store embeddings in Chroma
    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name=collection_name,
        persist_directory=VECTOR_STORE_DIR,
        ids=ids,
    )

    # Retriever
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 4
        }
    )

    return vectorstore, retriever