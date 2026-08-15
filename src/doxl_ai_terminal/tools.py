"""
Central tool registry for the AI agent.

Collects all @tool-decorated functions into categorized lists.
The AI agent (which the user will create) can import these directly::

    from doxl_ai_terminal.tools import ALL_TOOLS, load_file
    from doxl_ai_terminal.tools import DOC_TOOLS, EXCEL_TOOLS, VECTOR_TOOLS
"""

from __future__ import annotations

from typing import List

from langchain_core.tools import tool

# ── Document Manager Tools (via Frontier/manager_tools.py) ──

from doxl_ai_terminal.Frontier.manager_tools import (
    load_file,
    managers,
    # Doc tools
    doc_view,
    doc_search,
    doc_update_line,
    doc_add_line,
    doc_delete_line,
    doc_replace_all,
    doc_history,
    # Excel tools
    excel_view,
    excel_search,
    excel_update_cell,
    excel_add_cell,
    excel_add_row,
    excel_delete_cell,
    excel_delete_row,
    excel_replace_all,
    excel_history,
)

# ── Sentence Extraction (internal functions — we wrap them as tools below) ──

from doxl_ai_terminal.data_structure.full_sentence_extractor import (
    extract_full_sentences as _extract_full_sentences,
    extract_from_retrieved_chunk as _extract_from_chunk,
    extract_with_context as _extract_with_context,
)

# ── Vector DB Manager ──

from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager

# Create a shared VectorDBManager for tool functions
_vector_manager = VectorDBManager()


# ═══════════════════════════════════════════════════════
# Vector Tools — @tool wrappers around VectorDBManager
# ═══════════════════════════════════════════════════════

@tool
def vector_list_collections() -> str:
    """List all vector store collections."""
    collections = _vector_manager.list_collections()
    if not collections:
        return "No collections found."
    return "\n".join(f"  - {c}" for c in collections)


@tool
def vector_load_collection(collection_name: str) -> str:
    """Load a vector store collection for searching."""
    _vector_manager.load_collection(collection_name)
    count = _vector_manager.collection_count()
    return f"Loaded '{collection_name}' ({count} documents)"


@tool
def vector_search(query: str, k: int = 4) -> str:
    """Search the loaded vector store collection for relevant documents."""
    try:
        results = _vector_manager.search_with_scores(query, k=k)
    except RuntimeError as e:
        return str(e)

    if not results:
        return "No results found."

    lines = []
    for doc, score in results:
        lines.append(f"  [{score:.4f}] {doc.page_content[:120]}")
        lines.append(f"           metadata: {doc.metadata}")
    return "\n".join(lines)


@tool
def vector_delete_collection(collection_name: str) -> str:
    """Delete a vector store collection."""
    return _vector_manager.delete_collection(collection_name)


# ═══════════════════════════════════════════════════════
# Sentence Tools — @tool wrappers that access doc_data
#                  through the shared manager
# ═══════════════════════════════════════════════════════

@tool
def sentence_extract(query: str) -> str:
    """Find all complete sentences in the loaded Word document that contain the query."""
    if not managers["doc"]:
        return "No document loaded."
    results = _extract_full_sentences(managers["doc"].doc_data, query)
    if not results:
        return "No matching sentences found."
    lines = []
    for r in results:
        lines.append(f"  P{r.paragraph} S{r.sentence_index}: {r.sentence}")
    return "\n".join(lines)


@tool
def sentence_extract_from_chunk(chunk_text: str) -> str:
    """Given a retrieved chunk, find the complete sentence(s) from the loaded Word document."""
    if not managers["doc"]:
        return "No document loaded."
    results = _extract_from_chunk(managers["doc"].doc_data, chunk_text)
    if not results:
        return "No matching sentences found."
    lines = []
    for r in results:
        lines.append(f"  P{r.paragraph} S{r.sentence_index}: {r.sentence}")
    return "\n".join(lines)


@tool
def sentence_extract_with_context(chunk_text: str, context_window: int = 1) -> str:
    """Extract the full sentence + surrounding sentences from the loaded Word document."""
    if not managers["doc"]:
        return "No document loaded."
    results = _extract_with_context(managers["doc"].doc_data, chunk_text, context_window)
    if not results:
        return "No matching sentences found."
    lines = []
    for r in results:
        lines.append(f"  [P{r['paragraph']}] {r['full_context']}")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════
# Categorized Tool Lists
# ═══════════════════════════════════════════════════════

DOC_TOOLS: List = [
    doc_view,
    doc_search,
    doc_update_line,
    doc_add_line,
    doc_delete_line,
    doc_replace_all,
    doc_history,
]

EXCEL_TOOLS: List = [
    excel_view,
    excel_search,
    excel_update_cell,
    excel_add_cell,
    excel_add_row,
    excel_delete_cell,
    excel_delete_row,
    excel_replace_all,
    excel_history,
]

VECTOR_TOOLS: List = [
    vector_list_collections,
    vector_load_collection,
    vector_search,
    vector_delete_collection,
]

SENTENCE_TOOLS: List = [
    sentence_extract,
    sentence_extract_from_chunk,
    sentence_extract_with_context,
]

ALL_TOOLS: List = DOC_TOOLS + EXCEL_TOOLS + VECTOR_TOOLS + SENTENCE_TOOLS

__all__ = [
    "load_file",
    "DOC_TOOLS",
    "EXCEL_TOOLS",
    "VECTOR_TOOLS",
    "SENTENCE_TOOLS",
    "ALL_TOOLS",
    # Individual tools re-exported for convenience
    "doc_view", "doc_search", "doc_update_line", "doc_add_line",
    "doc_delete_line", "doc_replace_all", "doc_history",
    "excel_view", "excel_search", "excel_update_cell", "excel_add_cell",
    "excel_add_row", "excel_delete_cell", "excel_delete_row",
    "excel_replace_all", "excel_history",
    "vector_list_collections", "vector_load_collection",
    "vector_search", "vector_delete_collection",
    "sentence_extract", "sentence_extract_from_chunk",
    "sentence_extract_with_context",
]
