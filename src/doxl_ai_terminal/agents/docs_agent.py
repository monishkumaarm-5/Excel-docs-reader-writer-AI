"""
DOXL AI — LangGraph Multi-Agent Document Handler

Architecture:

    Code Router
        |
        +--> Reader
        |      ├── search_document
        |      ├── vector_search
        |      ├── find_full_sentence
        |      └── find_with_context
        |
        +--> Writer
        |      ├── add_line
        |      ├── batch_edit_document
        |      └── create_document_content
        |
        +--> Updater
        |      ├── update_line
        |      ├── replace_all
        |      └── batch_edit_document
        |
        +--> Deleter
               ├── delete_line
               └── batch_edit_document

Each specialist is a LangGraph subgraph.

Important behavior:

    EMPTY DOCUMENT + EXPLICIT ADD
        -> Generate requested content
        -> Actually write content
        -> Verify changelog
        -> Return VERIFIED

    EMPTY DOCUMENT + UPDATE/DELETE/SEARCH
        -> Explain there is no existing content
        -> Do not invent existing content

Usage:

    python docs_agent.py report.docx
"""

import os
import sys
import json
import re

from typing import List, Dict, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition

from langchain_core.tools import tool
from langchain_core.messages import (
    SystemMessage,
    HumanMessage,
    AIMessage,
)

# ------------------------------------------------------------------
# Your existing modules
# ------------------------------------------------------------------

from doxl_ai_terminal.Frontier.ChangeLogManager_docs import LiveDocManager
from doxl_ai_terminal.Frontier.fileReader import read_word

from doxl_ai_terminal.Chunker.chunking import (
    chunk_doc_sub_para,
    chunk_doc_line_by_line,
)

from doxl_ai_terminal.data_handler.vector_config import store_in_chroma, make_collection_name

from doxl_ai_terminal.data_handler.vector_db_operation import (
    VectorDBManager,
)

from doxl_ai_terminal.data_structure.full_sentence_extractor import (
    extract_full_sentences,
    extract_from_retrieved_chunk,
    extract_with_context,
)

from doxl_ai_terminal.pipeline.terminal_ui import (
    Spinner,
    success,
    info,
    warn,
    error,
    header,
    agent_result,
    prompt_input,
)


# ================================================================
# SECTION 1: SHARED STATE
# ================================================================

_state: Dict = {
    "doc_manager": None,
    "doc_data": None,
    "vector_managers": {},
    "collections": [],
    "filepath": None,
    "_dirty": False,
}


# ================================================================
# SECTION 2: FILE PROCESSOR
# ================================================================

def process_file(filepath: str):
    """
    Read the .docx, load the document manager and build vector indexes.

    Empty documents are valid.
    """

    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"File not found: {filepath}"
        )

    ext = os.path.splitext(filepath)[1].lower()

    if ext != ".docx":
        raise ValueError(
            f"Only .docx files supported. Got: {ext}"
        )

    _state["filepath"] = filepath

    with Spinner("Loading the file..."):

        # ----------------------------------------------------------
        # Step 1: Read document
        # ----------------------------------------------------------

        doc_data = read_word(filepath)

        _state["doc_data"] = doc_data

        # ----------------------------------------------------------
        # Step 2: Load document manager
        # ----------------------------------------------------------

        _state["doc_manager"] = LiveDocManager(filepath)

        # ----------------------------------------------------------
        # Step 3: Build vector index
        #
        # Empty documents simply produce no collections.
        # ----------------------------------------------------------

        _rebuild_doc_vector_index(
            filepath,
            doc_data,
        )

    success(
        f"Ready! "
        f"{doc_data.total_lines} lines, "
        f"{doc_data.total_paragraphs} paragraphs loaded.\n"
    )


# ================================================================
# SECTION 2B: VECTOR INDEX
# ================================================================

CHUNK_STRATEGIES = [
    (
        "line_by_line",
        chunk_doc_line_by_line,
    ),
    (
        "sub_para",
        chunk_doc_sub_para,
    ),
]


def _rebuild_doc_vector_index(
    filepath: str,
    doc_data,
) -> None:
    """
    Rebuild vector indexes for the current document.

    Empty documents simply result in no vector collections.
    """

    db = VectorDBManager()

    collections = []
    vector_managers = {}

    for format_name, chunk_fn in CHUNK_STRATEGIES:

        try:
            chunks = chunk_fn(doc_data)
        except Exception as e:
            warn(
                f"Chunking failed for {format_name}: {e}"
            )
            continue

        if not chunks:
            continue

        collection_name = make_collection_name(
            filepath,
            format_name,
        )

        try:
            db.delete_collection(collection_name)
        except Exception:
            pass

        store_in_chroma(
            chunks,
            collection_name,
        )

        collections.append(collection_name)

        vector_managers[format_name] = (
            VectorDBManager()
            .load_collection(collection_name)
        )

    _state["collections"] = collections
    _state["vector_managers"] = vector_managers


def _refresh_doc_data() -> None:
    """
    Re-read the current .docx from disk.

    This is used after a write operation.
    """

    filepath = _state.get("filepath")

    if not filepath:
        return

    _state["doc_data"] = read_word(filepath)

    _state["_dirty"] = True


def mark_doc_dirty_and_refresh() -> None:
    """
    Called after every mutation.
    """

    try:
        _refresh_doc_data()

    except Exception as e:
        warn(
            f"Refreshing document data failed: {e}"
        )


def rebuild_doc_vector_index_if_dirty() -> None:
    """
    Rebuild the expensive vector index once after a request.
    """

    if not _state.get("_dirty"):
        return

    filepath = _state.get("filepath")
    doc_data = _state.get("doc_data")

    if not filepath or doc_data is None:
        return

    try:

        _rebuild_doc_vector_index(
            filepath,
            doc_data,
        )

        _state["_dirty"] = False

    except Exception as e:

        warn(
            f"Vector index refresh failed: {e}"
        )


# ================================================================
# SECTION 2C: DOCUMENT STATE HELPERS
# ================================================================

def _document_is_empty() -> bool:
    """
    Determine whether the loaded document contains usable content.
    """

    doc_data = _state.get("doc_data")

    if doc_data is None:
        return True

    total_lines = getattr(
        doc_data,
        "total_lines",
        0,
    )

    total_paragraphs = getattr(
        doc_data,
        "total_paragraphs",
        0,
    )

    return (
        total_lines == 0
        and total_paragraphs == 0
    )


def _document_summary() -> str:
    """
    Return a concise document state description.
    """

    doc_data = _state.get("doc_data")

    if doc_data is None:
        return "No document data is loaded."

    lines = getattr(
        doc_data,
        "total_lines",
        0,
    )

    paragraphs = getattr(
        doc_data,
        "total_paragraphs",
        0,
    )

    if lines == 0 and paragraphs == 0:
        return "The document is completely empty."

    return (
        f"The document contains "
        f"{lines} lines and "
        f"{paragraphs} paragraphs."
    )


# ================================================================
# SECTION 2D: WRITE INTENT
# ================================================================

_WRITE_INTENT_KEYWORDS = (
    # Creation
    "add",
    "insert",
    "append",
    "write",
    "create",
    "populate",
    "fill",
    "generate",
    "compose",
    "draft",
    "include",
    "put",
    "introduce",

    # Update
    "update",
    "edit",
    "modify",
    "change",
    "replace",
    "fix",
    "correct",
    "rewrite",
    "rename",
    "refactor",

    # Delete
    "delete",
    "remove",
    "erase",

    # Formatting
    "space",
    "spacing",
    "blank",
    "gap",
    "format",
    "formatting",
    "indent",
    "align",
    "line break",
    "empty line",
)


def _contains_keyword(
    text: str,
    keyword: str,
) -> bool:

    pattern = rf"\b{re.escape(keyword)}\b"

    return bool(
        re.search(
            pattern,
            text.lower(),
        )
    )


def _looks_like_write_request(
    user_input: str,
) -> bool:

    return any(
        _contains_keyword(
            user_input,
            keyword,
        )
        for keyword in _WRITE_INTENT_KEYWORDS
    )


# ================================================================
# SECTION 3: READER TOOLS
# ================================================================

@tool
def search_document(keyword: str) -> str:
    """
    Search for an exact keyword in the Word document.

    Returns matching paragraph and line positions.
    """

    mgr = _state.get("doc_manager")

    if not mgr:
        return "ERROR: No document loaded."

    if _document_is_empty():
        return (
            "DOCUMENT EMPTY: "
            "The document contains no searchable content."
        )

    results = mgr.search(keyword)

    if not results:
        return (
            f"No matches found for '{keyword}'."
        )

    lines = [
        f"P{r.paragraph} L{r.line}: {r.line_str}"
        for r in results
    ]

    return "\n".join(lines)


MAX_RELEVANT_DISTANCE = 1.3


@tool
def vector_search(query: str) -> str:
    """
    Semantic search across the document.
    """

    managers = (
        _state.get("vector_managers")
        or {}
    )

    if not managers:

        if _document_is_empty():
            return (
                "DOCUMENT EMPTY: "
                "There is no document content to search."
            )

        return (
            "No vector store is currently available."
        )

    candidates = []

    for format_name, vm in managers.items():

        try:

            candidates.extend(
                vm.search_with_scores(
                    query,
                    k=4,
                )
            )

        except Exception as e:

            return (
                f"Search error "
                f"({format_name}): {e}"
            )

    relevant = [
        pair
        for pair in candidates
        if pair[1] <= MAX_RELEVANT_DISTANCE
    ]

    relevant.sort(
        key=lambda pair: pair[1]
    )

    top = relevant[:4]

    if not top:

        return (
            f"No semantic matches for "
            f"'{query}'."
        )

    output = []

    for i, (doc, score) in enumerate(
        top,
        1,
    ):

        para = doc.metadata.get(
            "paragraph",
            "?",
        )

        fmt = doc.metadata.get(
            "format",
            "?",
        )

        output.append(
            f"[{i}] "
            f"(Paragraph {para}, {fmt}) "
            f"{doc.page_content}"
        )

    return "\n".join(output)


@tool
def find_full_sentence(query: str) -> str:
    """
    Find complete sentences containing the query.
    """

    doc_data = _state.get(
        "doc_data"
    )

    if not doc_data:
        return (
            "ERROR: No document loaded."
        )

    if _document_is_empty():
        return (
            "DOCUMENT EMPTY: "
            "There are no sentences to search."
        )

    results = extract_full_sentences(
        doc_data,
        query,
    )

    if not results:
        return (
            f"No sentences containing "
            f"'{query}'."
        )

    output = []

    for r in results:

        output.append(
            f"P{r.paragraph} "
            f"Sentence {r.sentence_index}: "
            f"{r.sentence}"
        )

    return "\n".join(output)


@tool
def find_with_context(query: str) -> str:
    """
    Find matching sentences with surrounding context.
    """

    doc_data = _state.get(
        "doc_data"
    )

    if not doc_data:
        return (
            "ERROR: No document loaded."
        )

    if _document_is_empty():
        return (
            "DOCUMENT EMPTY: "
            "There is no context to search."
        )

    results = extract_with_context(
        doc_data,
        query,
        context_window=1,
    )

    if not results:
        return (
            f"No context found for "
            f"'{query}'."
        )

    output = []

    for r in results:

        output.append(
            f"--- Paragraph "
            f"{r['paragraph']} ---"
        )

        if r["before"]:

            output.append(
                f"  Before: "
                f"{' '.join(r['before'])}"
            )

        output.append(
            f"  >>> "
            f"{r['matched_sentence']}"
        )

        if r["after"]:

            output.append(
                f"  After: "
                f"{' '.join(r['after'])}"
            )

        output.append("")

    return "\n".join(output)


# ================================================================
# SECTION 4: WRITER TOOLS
# ================================================================

@tool
def create_document_content(text: str) -> str:
    """
    Create initial content in an empty document.

    This tool exists specifically for:

        empty document
            +
        explicit user request to create/add/write content

    The content is actually written to disk.

    Do NOT use this for updating existing content.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    if not _document_is_empty():
        return (
            "ERROR: The document is not empty. "
            "Use add_line or batch_edit_document "
            "for existing documents."
        )

    if not text or not text.strip():
        return (
            "ERROR: Cannot create empty content."
        )

    try:

        # ----------------------------------------------------------
        # Preferred implementation:
        #
        # If your LiveDocManager has a dedicated method,
        # use it.
        # ----------------------------------------------------------

        if hasattr(
            mgr,
            "create_initial_content",
        ):

            result = (
                mgr.create_initial_content(
                    text
                )
            )

        # ----------------------------------------------------------
        # Fallback:
        #
        # Use the existing add_line implementation.
        #
        # This requires LiveDocManager.add_line(1, 1, text)
        # to support an empty document.
        # ----------------------------------------------------------

        else:

            result = mgr.add_line(
                1,
                1,
                text,
            )

        # ----------------------------------------------------------
        # Refresh local state.
        # ----------------------------------------------------------

        mark_doc_dirty_and_refresh()

        return (
            "SUCCESS: Initial document "
            "content was written.\n"
            f"{result}"
        )

    except Exception as e:

        return (
            "ERROR: Failed to create "
            f"document content: {e}"
        )


@tool
def add_line(
    paragraph: int,
    line: int,
    text: str,
) -> str:
    """
    Add a line to an existing document.

    For an empty document, use create_document_content
    instead.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    if _document_is_empty():
        return (
            "ERROR: The document is empty. "
            "Use create_document_content "
            "for the initial content."
        )

    result = mgr.add_line(
        paragraph,
        line,
        text,
    )

    mark_doc_dirty_and_refresh()

    return result


@tool
def batch_edit_document(
    edits_json: str,
) -> str:
    """
    Apply multiple add/update/delete edits in one call.

    For an empty document, an explicit add request may use:

    [
        {
            "action": "add",
            "paragraph": 1,
            "line": 1,
            "text": "..."
        }
    ]

    If the manager does not support that operation,
    use create_document_content instead.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    try:

        edits = json.loads(
            edits_json
        )

    except json.JSONDecodeError as e:

        return (
            "ERROR: edits_json must be "
            "a valid JSON array. "
            f"{e}"
        )

    if not isinstance(
        edits,
        list,
    ) or not edits:

        return (
            "ERROR: edits_json must be "
            "a non-empty JSON array."
        )

    normalized = []

    for e in edits:

        if (
            not isinstance(e, dict)
            or "action" not in e
        ):

            return (
                "ERROR: Every edit needs "
                "an 'action' key."
            )

        action = str(
            e.get("action", "")
        ).lower()

        if action not in {
            "add",
            "update",
            "delete",
        }:

            return (
                f"ERROR: Unsupported "
                f"action '{action}'. "
                "Use add, update, or delete."
            )

        try:

            paragraph = int(
                e.get("paragraph")
            )

            line = int(
                e.get("line")
            )

        except (
            TypeError,
            ValueError,
        ):

            return (
                "ERROR: paragraph and "
                "line must be integers."
            )

        normalized.append(
            {
                "action": action,
                "paragraph": paragraph,
                "line": line,
                "text": e.get(
                    "text",
                    "",
                ),
            }
        )

    try:

        results = mgr.batch_edit(
            normalized
        )

        mark_doc_dirty_and_refresh()

        return (
            f"Applied {len(results)} "
            "edit(s) in one save:\n"
            + "\n".join(results)
        )

    except Exception as e:

        return (
            "ERROR: Batch edit failed: "
            f"{e}"
        )


# ================================================================
# SECTION 5: UPDATER TOOLS
# ================================================================

@tool
def update_line(
    paragraph: int,
    line: int,
    new_text: str,
) -> str:
    """
    Update an existing line.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    if _document_is_empty():
        return (
            "ERROR: Cannot update an empty "
            "document because there is no "
            "existing content."
        )

    result = mgr.update_line(
        paragraph,
        line,
        new_text,
    )

    mark_doc_dirty_and_refresh()

    return result


@tool
def replace_all(
    old_text: str,
    new_text: str,
) -> str:
    """
    Replace text across the entire document.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    if _document_is_empty():
        return (
            "ERROR: Cannot replace content "
            "because the document is empty."
        )

    result = mgr.replace_all(
        old_text,
        new_text,
    )

    mark_doc_dirty_and_refresh()

    return result


# ================================================================
# SECTION 6: DELETER
# ================================================================

@tool
def delete_line(
    paragraph: int,
    line: int,
) -> str:
    """
    Delete a specific line.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    if _document_is_empty():
        return (
            "ERROR: Cannot delete content "
            "because the document is empty."
        )

    result = mgr.delete_line(
        paragraph,
        line,
    )

    mark_doc_dirty_and_refresh()

    return result


# ================================================================
# SECTION 7: SHARED TOOLS
# ================================================================

@tool
def show_history(
    check: str = "all",
) -> str:
    """
    Show document change history.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    return mgr.history()


@tool
def save_copy(
    output_path: str,
) -> str:
    """
    Save the current document to a new file.
    """

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    return mgr.save_as(
        output_path
    )


@tool
def reload_from_disk(
    confirm: str = "yes",
) -> str:
    """
    Reload document from disk.
    """

    if confirm.lower() != "yes":

        return (
            "Reload cancelled. "
            "Pass confirm='yes' to proceed."
        )

    mgr = _state.get(
        "doc_manager"
    )

    if not mgr:
        return (
            "ERROR: No document loaded."
        )

    result = mgr.reload()

    try:
        _state["doc_data"] = read_word(
            _state["filepath"]
        )

        _state["_dirty"] = False

    except Exception:
        pass

    return result


# ================================================================
# SECTION 8: DOCUMENT DATA MODEL
# ================================================================

_DOC_DATA_MODEL = (
    "\n\nDOCUMENT DATA MODEL:\n"

    "- The document is represented as a flat list of lines. "
    "Each line is addressed by paragraph number (P) and line "
    "number within that paragraph (L).\n"

    "- P numbers come from the Word document's real paragraph "
    "order and CAN skip numbers because blank paragraphs may "
    "exist for spacing.\n"

    "- Never invent an existing P/L address.\n"

    "- For an EXISTING document, search first to determine "
    "the correct P/L address.\n"

    "- L numbers are 1-indexed.\n"

    "- Adding a line shifts later lines in the same paragraph.\n"

    "- If multiple lines are added to the same paragraph, "
    "use batch_edit_document and list additions in the "
    "correct insertion order.\n"

    "- If the document is completely empty, there is no "
    "existing P/L address to search for. Treat it as a "
    "special creation case.\n"
)


# ================================================================
# SECTION 8A: DATA SUFFICIENCY POLICY
# ================================================================

_DATA_SUFFICIENCY_POLICY = (
    "\n\nDATA SUFFICIENCY AND WRITE POLICY:\n"

    "IMPORTANT DISTINCTION:\n"

    "There is a difference between:\n"
    "1. Inventing or pretending that information already exists "
    "in the document.\n"
    "2. Generating NEW information that the user explicitly "
    "asked you to add.\n\n"

    "The first is prohibited.\n"
    "The second is allowed and REQUIRED when the user explicitly "
    "requests new content.\n\n"

    "EMPTY DOCUMENT:\n"

    "- An empty document does NOT prevent an explicit ADD, CREATE, "
    "WRITE, INSERT, POPULATE, or FILL request.\n"

    "- If the user explicitly requests new content, generate "
    "appropriate content for exactly what they requested and "
    "actually write it to the document.\n"

    "- Do NOT repeatedly search an empty document.\n"

    "- Do NOT ask the user to provide the content again when "
    "the user has already explicitly requested what content "
    "should be created.\n"

    "- Do NOT say that you are prohibited from generating the "
    "requested information merely because the document is empty.\n"

    "- For an empty document, use create_document_content.\n"

    "EMPTY DOCUMENT + READ:\n"

    "- If the user asks to find/search/read existing information "
    "from an empty document, explain that no information exists.\n"

    "EMPTY DOCUMENT + UPDATE:\n"

    "- If the user asks to update/modify/replace existing content "
    "in an empty document, explain that there is no existing "
    "content to update.\n"

    "EMPTY DOCUMENT + DELETE:\n"

    "- If the user asks to delete existing content from an empty "
    "document, explain that there is nothing to delete.\n"

    "NON-EMPTY DOCUMENT:\n"

    "- Do not invent existing document content.\n"

    "- For update/delete operations, search for the actual "
    "content first.\n"

    "- If existing content is insufficient for an UPDATE or "
    "DELETE request, explain exactly what is missing.\n"

    "- However, an explicit request to ADD new information "
    "may still be fulfilled even when the existing document "
    "has little or unrelated content.\n"

    "WRITE REQUIREMENT:\n"

    "- Never claim a document was modified unless an actual "
    "write tool was called successfully.\n"

    "- Writing text in the final response does NOT modify "
    "the file.\n"

    "- The actual file must be changed through a write tool.\n"
)


# ================================================================
# SECTION 8B: SPECIALIST PROMPTS
# ================================================================

_READER_PROMPT = (
    "ROLE: Document Reader\n"

    "GOAL: Search and understand existing document content "
    "with precision.\n\n"

    "You are the document research specialist.\n"

    "Use search_document for exact text.\n"
    "Use vector_search for semantic meaning.\n"
    "Use find_full_sentence when complete sentences are needed.\n"
    "Use find_with_context when surrounding context is needed.\n\n"

    "If the document is empty, do not pretend that content exists. "
    "Tell the user that the document has no searchable information.\n"

    + _DOC_DATA_MODEL
    + _DATA_SUFFICIENCY_POLICY
)


_READER_TOOLS = [
    search_document,
    vector_search,
    find_full_sentence,
    find_with_context,
]


_WRITER_PROMPT = (
    "ROLE: Document Writer\n"

    "GOAL: Create and add new content to the document.\n\n"

    "You are responsible for ACTUAL document writes.\n\n"

    "CRITICAL RULE:\n"

    "If the user explicitly asks you to add, create, write, "
    "insert, populate, fill, generate, compose, or append "
    "information, you MUST perform the actual write operation.\n\n"

    "EMPTY DOCUMENT:\n"

    "If the document is empty and the user explicitly asks "
    "for new content:\n"

    "1. Understand what the user requested.\n"
    "2. Generate appropriate content for that request.\n"
    "3. Call create_document_content with the generated content.\n"
    "4. Verify that the tool returned success.\n"
    "5. Only then report that the content was added.\n\n"

    "DO NOT search an empty document before creating content.\n"

    "DO NOT ask the user to provide content again when the "
    "user already clearly specified what they want written.\n\n"

    "NON-EMPTY DOCUMENT:\n"

    "If the document already contains content, use search_document "
    "or vector_search to determine where the new content belongs.\n\n"

    "MULTI-LINE CONTENT:\n"

    "For large additions to an existing document, prefer "
    "batch_edit_document.\n\n"

    "ACTUAL WRITE REQUIREMENT:\n"

    "The final response does NOT save anything.\n"

    "You are NOT finished until the appropriate write tool "
    "has actually been called.\n"

    + _DOC_DATA_MODEL
    + _DATA_SUFFICIENCY_POLICY
)


_WRITER_TOOLS = [
    search_document,
    vector_search,
    add_line,
    batch_edit_document,
    create_document_content,
    show_history,
]


_UPDATER_PROMPT = (
    "ROLE: Document Updater\n"

    "GOAL: Modify existing document content accurately.\n\n"

    "Workflow:\n"

    "1. Search for the content that needs modification.\n"
    "2. Identify the exact P/L address.\n"
    "3. Perform the update.\n"
    "4. Verify the result.\n\n"

    "If the document is empty, there is no existing content "
    "to update. Do not invent existing content.\n\n"

    "For global text changes use replace_all.\n"

    "For multiple changes use batch_edit_document.\n\n"

    "The final response is not the modification. "
    "An actual update tool must be called.\n"

    + _DOC_DATA_MODEL
    + _DATA_SUFFICIENCY_POLICY
)


_UPDATER_TOOLS = [
    search_document,
    vector_search,
    update_line,
    replace_all,
    batch_edit_document,
    show_history,
]


_DELETER_PROMPT = (
    "ROLE: Document Deleter\n"

    "GOAL: Safely remove existing document content.\n\n"

    "Deletion workflow:\n"

    "1. Search for the content.\n"
    "2. Confirm the exact P/L address.\n"
    "3. Delete the content.\n"
    "4. Verify the deletion.\n\n"

    "If the document is empty, there is nothing to delete. "
    "Do not invent content or P/L addresses.\n\n"

    "For multiple deletions, use batch_edit_document.\n\n"

    "The final answer does not delete anything. "
    "A delete tool must actually be called.\n"

    + _DOC_DATA_MODEL
    + _DATA_SUFFICIENCY_POLICY
)


_DELETER_TOOLS = [
    search_document,
    vector_search,
    delete_line,
    batch_edit_document,
    show_history,
]


_SPECIALIST_DEFS = {
    "reader": (
        _READER_PROMPT,
        _READER_TOOLS,
    ),

    "writer": (
        _WRITER_PROMPT,
        _WRITER_TOOLS,
    ),

    "updater": (
        _UPDATER_PROMPT,
        _UPDATER_TOOLS,
    ),

    "deleter": (
        _DELETER_PROMPT,
        _DELETER_TOOLS,
    ),
}


# ================================================================
# SECTION 9: SPECIALIST SUBGRAPHS
# ================================================================

class SpecialistState(TypedDict):
    messages: Annotated[
        list,
        add_messages,
    ]


def _build_specialist_subgraph(
    node_name: str,
    system_prompt: str,
    tools: list,
    llm,
):

    llm_with_tools = llm.bind_tools(
        tools
    )

    def agent_node(
        state: SpecialistState,
    ) -> SpecialistState:

        response = (
            llm_with_tools.invoke(
                [
                    SystemMessage(
                        content=system_prompt
                    )
                ]
                + state["messages"]
            )
        )

        return {
            "messages": [
                response
            ]
        }

    graph = StateGraph(
        SpecialistState
    )

    graph.add_node(
        "agent",
        agent_node,
    )

    graph.add_node(
        "tools",
        ToolNode(tools),
    )

    graph.add_edge(
        START,
        "agent",
    )

    graph.add_conditional_edges(
        "agent",
        tools_condition,
        {
            "tools": "tools",
            END: END,
        },
    )

    graph.add_edge(
        "tools",
        "agent",
    )

    return graph.compile(
        name=node_name
    )


def build_specialists(llm) -> Dict:

    return {
        name: _build_specialist_subgraph(
            name,
            prompt,
            tools,
            llm,
        )
        for name, (
            prompt,
            tools,
        ) in _SPECIALIST_DEFS.items()
    }


# ================================================================
# SECTION 10: ROUTER
# ================================================================

_INTENT_ROUTES = [

    # Delete first because:
    #
    # "delete old pricing and add new pricing"
    #
    # should remove old information before adding new information.
    (
        "deleter",
        (
            "delete",
            "remove",
            "erase",
        ),
    ),

    (
        "updater",
        (
            "update",
            "edit",
            "modify",
            "change",
            "replace",
            "fix",
            "correct",
            "rewrite",
            "rename",
            "refactor",
            "space",
            "spacing",
            "blank",
            "gap",
            "format",
            "formatting",
            "indent",
            "align",
            "line break",
            "empty line",
        ),
    ),

    (
        "writer",
        (
            "add",
            "insert",
            "append",
            "continue",
            "write",
            "create",
            "populate",
            "fill",
            "generate",
            "compose",
            "draft",
            "include",
            "put",
            "introduce",
        ),
    ),
]


def route_agents(
    user_request: str,
) -> List[str]:
    """
    Determine which specialists should run.

    No LLM router is used.
    """

    text = user_request.lower()

    matched = []

    for name, keywords in _INTENT_ROUTES:

        if any(
            _contains_keyword(
                text,
                keyword,
            )
            for keyword in keywords
        ):

            matched.append(name)

    if matched:
        return matched

    if _looks_like_write_request(
        user_request
    ):

        return ["updater"]

    return ["reader"]


# ================================================================
# SECTION 11: CREW STATE
# ================================================================

class CrewState(TypedDict):

    messages: Annotated[
        list,
        add_messages,
    ]

    pending: List[str]


def _route_node(
    state: CrewState,
) -> CrewState:

    last_human = next(
        (
            m
            for m in reversed(
                state["messages"]
            )
            if isinstance(
                m,
                HumanMessage,
            )
        ),
        None,
    )

    text = (
        last_human.content
        if last_human
        else ""
    )

    if not isinstance(
        text,
        str,
    ):

        text = str(text)

    return {
        "pending": route_agents(
            text
        )
    }


def _advance_node(
    state: CrewState,
) -> CrewState:

    return {
        "pending": (
            state["pending"][1:]
        )
    }


def _dispatch(
    state: CrewState,
) -> str:

    pending = (
        state.get("pending")
        or []
    )

    return (
        pending[0]
        if pending
        else END
    )


# ================================================================
# SECTION 12: BUILD CREW GRAPH
# ================================================================

def build_crew_graph(
    specialists: Dict,
):

    graph = StateGraph(
        CrewState
    )

    graph.add_node(
        "route",
        _route_node,
    )

    graph.add_node(
        "advance",
        _advance_node,
    )

    for name, subgraph in specialists.items():

        graph.add_node(
            name,
            subgraph,
        )

    dispatch_map = {
        name: name
        for name in specialists
    }

    dispatch_map[END] = END

    graph.add_edge(
        START,
        "route",
    )

    graph.add_conditional_edges(
        "route",
        _dispatch,
        dispatch_map,
    )

    for name in specialists:

        graph.add_edge(
            name,
            "advance",
        )

    graph.add_conditional_edges(
        "advance",
        _dispatch,
        dispatch_map,
    )

    return graph.compile()


# ================================================================
# SECTION 13: RESPONSE HELPERS
# ================================================================

def _extract_text(
    content,
) -> str:

    if isinstance(
        content,
        str,
    ):

        return content

    if isinstance(
        content,
        list,
    ):

        parts = []

        for part in content:

            if (
                isinstance(
                    part,
                    dict,
                )
                and "text" in part
            ):

                parts.append(
                    part["text"]
                )

            elif isinstance(
                part,
                str,
            ):

                parts.append(part)

        return "\n".join(parts)

    return str(content)


def _last_answer_text(
    messages: list,
) -> str:
    """
    Return the last AI message containing visible text.
    """

    for m in reversed(
        messages
    ):

        if isinstance(
            m,
            AIMessage,
        ):

            text = _extract_text(
                m.content
            ).strip()

            if text:
                return text

    return "(no response)"


# ================================================================
# SECTION 14: MAIN SYSTEM
# ================================================================

class DocsAgentSystem:

    def __init__(
        self,
        filepath: str,
        llm=None,
    ):

        self.filepath = filepath

        # ----------------------------------------------------------
        # LLM
        # ----------------------------------------------------------

        if llm:

            self.llm = llm

        else:

            from doxl_ai_terminal.pipeline.config import (
                get_agentic_llm,
            )

            from doxl_ai_terminal.pipeline.credential_store import (
                load_credentials,
            )

            cred = load_credentials()

            if not cred:

                raise RuntimeError(
                    "No saved Gemini credentials found. "
                    "Run onboarding first or pass llm=."
                )

            info(
                "Using saved credentials "
                f"(model: {cred['model']})"
            )

            self.llm = get_agentic_llm(
                cred["api_key"],
                cred["model"],
            )

        # ----------------------------------------------------------
        # Load document
        # ----------------------------------------------------------

        header(
            "DOXL AI — Document Agent"
        )

        process_file(
            filepath
        )

        # ----------------------------------------------------------
        # Build graphs once
        # ----------------------------------------------------------

        self.specialists = (
            build_specialists(
                self.llm
            )
        )

        self.graph = (
            build_crew_graph(
                self.specialists
            )
        )

        # ----------------------------------------------------------
        # UI
        # ----------------------------------------------------------

        header(
            "Ready",
            "Reader · Writer · Updater · Deleter",
        )

        print(
            "  Ask in plain English, "
            'e.g. "find mentions of revenue".\n'
            "  'history' shows changes made, "
            "'quit' exits.\n"
            "  Full-document 'view' is disabled — "
            "reports can run 100+ pages."
        )


    # ============================================================
    # REQUEST PROCESSING
    # ============================================================

    def process_request(
        self,
        user_input: str,
    ) -> str:

        mgr = _state.get(
            "doc_manager"
        )

        changes_before = (
            len(
                mgr.changelog.changes
            )
            if mgr
            else 0
        )

        # ----------------------------------------------------------
        # Execute graph
        # ----------------------------------------------------------

        result_state = self.graph.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=user_input
                    )
                ],
                "pending": [],
            },
            config={
                "recursion_limit": 50
            },
        )

        # ----------------------------------------------------------
        # Rebuild vector index once
        # ----------------------------------------------------------

        rebuild_doc_vector_index_if_dirty()

        # ----------------------------------------------------------
        # Ground truth verification
        # ----------------------------------------------------------

        mgr = _state.get(
            "doc_manager"
        )

        changes_after = (
            len(
                mgr.changelog.changes
            )
            if mgr
            else 0
        )

        made = (
            changes_after
            - changes_before
        )

        result_str = _last_answer_text(
            result_state["messages"]
        )

        # ----------------------------------------------------------
        # Actual write happened
        # ----------------------------------------------------------

        if made > 0:

            result_str += (
                "\n\n"
                f"[VERIFIED] "
                f"{made} change(s) were "
                "actually written to the file."
            )

        # ----------------------------------------------------------
        # User clearly requested writing,
        # but no write tool executed.
        # ----------------------------------------------------------

        elif _looks_like_write_request(
            user_input
        ):

            result_str += (
                "\n\n"
                "[WARNING] This request appears "
                "to require a document modification, "
                "but no write tool was actually called. "
                "The file was NOT modified."
            )

        return result_str


    # ============================================================
    # CHAT LOOP
    # ============================================================

    def chat(self):

        while True:

            try:

                user_input = prompt_input(
                    "\nYou: "
                )

            except (
                KeyboardInterrupt,
                EOFError,
            ):

                print()

                success(
                    "Goodbye!"
                )

                break

            if not user_input:
                continue

            normalized = (
                user_input
                .strip()
                .lower()
            )

            # ------------------------------------------------------
            # Quit
            # ------------------------------------------------------

            if normalized in (
                "quit",
                "exit",
                "q",
                "bye",
            ):

                success(
                    "Goodbye!"
                )

                break

            # ------------------------------------------------------
            # View disabled
            # ------------------------------------------------------

            if normalized == "view":

                warn(
                    "'view' is disabled — "
                    "a document can run 100+ pages. "
                    "Try searching naturally instead, "
                    "e.g. 'find the section about pricing'."
                )

                continue

            # ------------------------------------------------------
            # History shortcut
            # ------------------------------------------------------

            if normalized == "history":

                print(
                    show_history.invoke(
                        "all"
                    )
                )

                continue

            # ------------------------------------------------------
            # Request
            # ------------------------------------------------------

            spinner = Spinner(
                "Thinking..."
            ).start()

            try:

                result = (
                    self.process_request(
                        user_input
                    )
                )

                spinner.stop(
                    ok=False
                )

                agent_result(
                    result
                )

            except Exception as e:

                spinner.stop(
                    ok=False
                )

                error(
                    str(e)
                )

                info(
                    "Try rephrasing your request."
                )


# ================================================================
# SECTION 15: ENTRY POINT
# ================================================================

def main():

    if len(sys.argv) < 2:

        print(
            "Usage: "
            "python docs_agent.py "
            "<filepath.docx>"
        )

        print(
            "Example: "
            "python docs_agent.py report.docx"
        )

        sys.exit(1)

    filepath = sys.argv[1]

    # --------------------------------------------------------------
    # None = use saved Gemini credentials
    # --------------------------------------------------------------

    llm = None

    system = DocsAgentSystem(
        filepath,
        llm=llm,
    )

    system.chat()


# ================================================================
# RUN
# ================================================================

if __name__ == "__main__":
    main()