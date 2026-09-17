"""
DOXL AI — LangGraph Multi-Agent Document Handler

Architecture:
    Code Router
        ├── Reader   (search_document, vector_search, find_full_sentence, find_with_context)
        ├── Writer   (add_line, batch_edit_document, create_document_content)
        ├── Updater  (update_line, replace_all, batch_edit_document)
        └── Deleter  (delete_line, batch_edit_document)

Each specialist is a LangGraph subgraph.  Keyword-based routing (no LLM call)
picks which specialists run for a given user request.

Usage:
    python docs_agent.py report.docx
"""

import os
import sys
import json
import re
from collections import deque
from typing import List, Dict, Optional, TypedDict, Annotated
from pydantic import Field

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from doxl_ai_terminal.Frontier.ChangeLogManager_docs import LiveDocManager
from doxl_ai_terminal.Frontier.fileReader import read_word
from doxl_ai_terminal.Chunker.chunking import chunk_doc_sub_para, chunk_doc_line_by_line
from doxl_ai_terminal.data_handler.vector_config import store_in_chroma, make_collection_name
from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager
from doxl_ai_terminal.data_structure.full_sentence_extractor import (
    extract_full_sentences, extract_from_retrieved_chunk, extract_with_context,
)
from doxl_ai_terminal.pipeline.terminal_ui import (
    Spinner, TaskProgress, success, info, warn, error, header, agent_result, prompt_input,
)


# ── Shared state ─────────────────────────────────────────────────

_state: Dict = {
    "doc_manager": None,
    "doc_data": None,
    "vector_managers": {},
    "collections": [],
    "filepath": None,
    "_dirty": False,
    "task_queue": deque(),
}


# ── File processor ───────────────────────────────────────────────

def process_file(filepath: str):
    """Read the .docx, load the document manager and build vector indexes."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".docx":
        raise ValueError(f"Only .docx files supported. Got: {ext}")

    _state["filepath"] = filepath
    _state["task_queue"] = deque()

    with Spinner("Loading the file..."):
        doc_data = read_word(filepath)
        _state["doc_data"] = doc_data
        _state["doc_manager"] = LiveDocManager(filepath)
        _rebuild_doc_vector_index(filepath, doc_data)

    success(f"Ready! {doc_data.total_lines} lines, {doc_data.total_paragraphs} paragraphs loaded.\n")


# ── Vector index ─────────────────────────────────────────────────

CHUNK_STRATEGIES = [
    ("line_by_line", chunk_doc_line_by_line),
    ("sub_para", chunk_doc_sub_para),
]


def _rebuild_doc_vector_index(filepath: str, doc_data) -> None:
    """Rebuild vector indexes for the current document.  Empty docs produce no collections."""
    db = VectorDBManager()
    collections = []
    vector_managers = {}

    for format_name, chunk_fn in CHUNK_STRATEGIES:
        try:
            chunks = chunk_fn(doc_data)
        except Exception as e:
            warn(f"Chunking failed for {format_name}: {e}")
            continue
        if not chunks:
            continue

        collection_name = make_collection_name(filepath, format_name)
        try:
            db.delete_collection(collection_name)
        except Exception:
            pass

        store_in_chroma(chunks, collection_name)
        collections.append(collection_name)
        vector_managers[format_name] = VectorDBManager().load_collection(collection_name)

    _state["collections"] = collections
    _state["vector_managers"] = vector_managers


def _refresh_doc_data() -> None:
    """Sync _state["doc_data"] from the live LiveDocManager instead of
    re-reading the .docx from disk (which would be stale mid-request)."""
    mgr = _state.get("doc_manager")
    if mgr is None:
        return
    _state["doc_data"] = mgr.doc_data
    _state["_dirty"] = True


def mark_doc_dirty_and_refresh() -> None:
    """Called after every mutation."""
    try:
        _refresh_doc_data()
    except Exception as e:
        warn(f"Refreshing document data failed: {e}")


_task_progress: TaskProgress | None = None


def _queue_task(description: str) -> None:
    """Record one pending edit and update the live progress bar."""
    global _task_progress
    queue = _state.setdefault("task_queue", deque())
    queue.append(description)
    if _task_progress is None:
        _task_progress = TaskProgress("Edits")
    _task_progress.add(description)


def flush_task_queue() -> None:
    """Drain the pending-edit queue, save once, and close the progress bar."""
    global _task_progress
    queue = _state.get("task_queue")
    if not queue:
        if _task_progress:
            _task_progress.finish(saved=False)
            _task_progress = None
        return
    while queue:
        queue.popleft()

    mgr = _state.get("doc_manager")
    if mgr is not None:
        mgr._sync_to_file()

    if _task_progress:
        _task_progress.finish(saved=True)
        _task_progress = None


def rebuild_doc_vector_index_if_dirty() -> None:
    """Rebuild the expensive vector index once after a request."""
    if not _state.get("_dirty"):
        return
    filepath = _state.get("filepath")
    doc_data = _state.get("doc_data")
    if not filepath or doc_data is None:
        return
    try:
        _rebuild_doc_vector_index(filepath, doc_data)
        _state["_dirty"] = False
    except Exception as e:
        warn(f"Vector index refresh failed: {e}")


# ── Document state helpers ───────────────────────────────────────

def _document_is_empty() -> bool:
    doc_data = _state.get("doc_data")
    if doc_data is None:
        return True
    return getattr(doc_data, "total_lines", 0) == 0 and getattr(doc_data, "total_paragraphs", 0) == 0


def _document_summary() -> str:
    doc_data = _state.get("doc_data")
    if doc_data is None:
        return "No document data is loaded."
    lines = getattr(doc_data, "total_lines", 0)
    paragraphs = getattr(doc_data, "total_paragraphs", 0)
    if lines == 0 and paragraphs == 0:
        return "The document is completely empty."
    return f"The document contains {lines} lines and {paragraphs} paragraphs."


# ── Write intent detection ───────────────────────────────────────

_WRITE_INTENT_KEYWORDS = (
    "add", "insert", "append", "write", "create", "populate", "fill",
    "generate", "compose", "draft", "include", "put", "introduce",
    "update", "edit", "modify", "change", "replace", "fix", "correct",
    "rewrite", "rename", "refactor",
    "delete", "remove", "erase",
    "space", "spacing", "blank", "gap", "format", "formatting",
    "indent", "align", "line break", "empty line",
)


def _contains_keyword(text: str, keyword: str) -> bool:
    return bool(re.search(rf"\b{re.escape(keyword)}\b", text.lower()))


def _looks_like_write_request(user_input: str) -> bool:
    return any(_contains_keyword(user_input, kw) for kw in _WRITE_INTENT_KEYWORDS)


# ── Formatting kwargs helper ─────────────────────────────────────

_FMT_FIELDS = (
    "font_name", "font_size", "bold", "italic", "underline", "strikethrough",
    "font_color", "highlight_color", "superscript", "subscript", "alignment", "style",
)

_FMT_DOCSTRING = (
    "Optional formatting (leave any unset/None to keep unchanged): "
    "font_name, font_size (pt), bold, italic, underline, strikethrough, "
    'font_color (hex RGB, no \'#\'), highlight_color ("yellow"/"bright_green"/'
    '"turquoise"/"pink"/"blue"/"red"/"dark_blue"/"teal"/"green"/"violet"/'
    '"dark_red"/"dark_yellow"/"gray_50"/"gray_25"/"black"/"white"/"none"), '
    "superscript, subscript, alignment (left/center/right/justify), "
    'style (paragraph style, e.g. "Heading 1".."Heading 9", "Title", "Normal").'
)

# Pydantic Field descriptors for formatting kwargs — gives Gemini per-param descriptions
def _fmt_font_name():     return Field(default=None, description="Font family name, e.g. 'Arial', 'Times New Roman', 'Calibri'")
def _fmt_font_size():     return Field(default=None, description="Font size in points, e.g. 10.0, 12.0, 14.0, 24.0")
def _fmt_bold():          return Field(default=None, description="True = make bold, False = remove bold")
def _fmt_italic():        return Field(default=None, description="True = make italic, False = remove italic")
def _fmt_underline():     return Field(default=None, description="True = underline, False = remove underline")
def _fmt_strikethrough(): return Field(default=None, description="True = strikethrough, False = remove")
def _fmt_font_color():    return Field(default=None, description="Font color as hex RGB string without '#', e.g. 'FF0000' for red, '0000FF' for blue")
def _fmt_highlight():     return Field(default=None, description="Highlight color: yellow, bright_green, turquoise, pink, blue, red, dark_blue, teal, green, violet, dark_red, dark_yellow, gray_50, gray_25, black, white, none")
def _fmt_superscript():   return Field(default=None, description="True = superscript, False = remove")
def _fmt_subscript():     return Field(default=None, description="True = subscript, False = remove")
def _fmt_alignment():     return Field(default=None, description="Paragraph alignment: 'left', 'center', 'right', or 'justify'")
def _fmt_style():         return Field(default=None, description="Paragraph style: 'Normal', 'Heading 1' through 'Heading 9', 'Title'")


def _collect_fmt(**kwargs) -> dict:
    return {k: v for k, v in kwargs.items() if k in _FMT_FIELDS and v is not None}


# ═════════════════════════════════════════════════════════════════
#  READER TOOLS
# ═════════════════════════════════════════════════════════════════

@tool
def search_document(keyword: str) -> str:
    """Search for an exact keyword in the Word document.  Returns matching P/L positions."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "DOCUMENT EMPTY: The document contains no searchable content."
    results = mgr.search(keyword)
    if not results:
        return f"No matches found for '{keyword}'."
    return "\n".join(f"P{r.paragraph} L{r.line}: {r.line_str}" for r in results)


MAX_RELEVANT_DISTANCE = 1.3


@tool
def vector_search(query: str) -> str:
    """Semantic search across the document."""
    managers = _state.get("vector_managers") or {}
    if not managers:
        if _document_is_empty():
            return "DOCUMENT EMPTY: There is no document content to search."
        return "No vector store is currently available."

    candidates = []
    for format_name, vm in managers.items():
        try:
            candidates.extend(vm.search_with_scores(query, k=4))
        except Exception as e:
            return f"Search error ({format_name}): {e}"

    relevant = sorted(
        [p for p in candidates if p[1] <= MAX_RELEVANT_DISTANCE],
        key=lambda p: p[1],
    )[:4]

    if not relevant:
        return f"No semantic matches for '{query}'."

    output = []
    for i, (doc, score) in enumerate(relevant, 1):
        para = doc.metadata.get("paragraph", "?")
        fmt = doc.metadata.get("format", "?")
        output.append(f"[{i}] (Paragraph {para}, {fmt}) {doc.page_content}")
    return "\n".join(output)


@tool
def find_full_sentence(query: str) -> str:
    """Find complete sentences containing the query."""
    doc_data = _state.get("doc_data")
    if not doc_data:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "DOCUMENT EMPTY: There are no sentences to search."
    results = extract_full_sentences(doc_data, query)
    if not results:
        return f"No sentences containing '{query}'."
    return "\n".join(f"P{r.paragraph} Sentence {r.sentence_index}: {r.sentence}" for r in results)


@tool
def find_with_context(query: str) -> str:
    """Find matching sentences with surrounding context."""
    doc_data = _state.get("doc_data")
    if not doc_data:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "DOCUMENT EMPTY: There is no context to search."
    results = extract_with_context(doc_data, query, context_window=1)
    if not results:
        return f"No context found for '{query}'."

    output = []
    for r in results:
        output.append(f"--- Paragraph {r['paragraph']} ---")
        if r["before"]:
            output.append(f"  Before: {' '.join(r['before'])}")
        output.append(f"  >>> {r['matched_sentence']}")
        if r["after"]:
            output.append(f"  After: {' '.join(r['after'])}")
        output.append("")
    return "\n".join(output)


@tool
def get_paragraph_style(paragraph: int) -> str:
    """Inspect a paragraph's current style and heading level.
    Check this BEFORE calling format_line/add_line/update_line with a style value."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    style_info = mgr.get_paragraph_style(paragraph)
    if style_info is None:
        return f"Not found: no tracked content in paragraph {paragraph}."
    level = style_info["heading_level"]
    level_desc = "not a heading" if level is None else f"heading_level {level}"
    return f"P{paragraph}: style = {style_info['style']!r} ({level_desc})"


@tool
def get_document_outline(check: str = "all") -> str:
    """Show the document's heading structure (every Heading N/Title paragraph, indented by level)."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    outline = mgr.get_outline()
    if not outline:
        return "No headings found in this document."
    lines = ["Document outline:"]
    for entry in outline:
        indent = "  " * entry["level"]
        lines.append(f"{indent}L{entry['level']} (P{entry['paragraph']}): {entry['text']}")
    return "\n".join(lines)


# ═════════════════════════════════════════════════════════════════
#  WRITER TOOLS
# ═════════════════════════════════════════════════════════════════

@tool
def create_document_content(
    text: str,
    font_name: Optional[str] = _fmt_font_name(), font_size: Optional[float] = _fmt_font_size(),
    bold: Optional[bool] = _fmt_bold(), italic: Optional[bool] = _fmt_italic(),
    underline: Optional[bool] = _fmt_underline(), strikethrough: Optional[bool] = _fmt_strikethrough(),
    font_color: Optional[str] = _fmt_font_color(), highlight_color: Optional[str] = _fmt_highlight(),
    superscript: Optional[bool] = _fmt_superscript(), subscript: Optional[bool] = _fmt_subscript(),
    alignment: Optional[str] = _fmt_alignment(), style: Optional[str] = _fmt_style(),
) -> str:
    """Create initial content in an empty document. Do NOT use for updating existing content.
    All formatting params (font_size, bold, italic, etc.) are optional — set any to apply formatting."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    if not _document_is_empty():
        return "ERROR: The document is not empty. Use add_line or batch_edit_document for existing documents."
    if not text or not text.strip():
        return "ERROR: Cannot create empty content."

    try:
        if hasattr(mgr, "create_initial_content"):
            result = mgr.create_initial_content(text)
        else:
            fmt = _collect_fmt(**{k: v for k, v in locals().items() if k in _FMT_FIELDS})
            result = mgr.add_line(1, 1, text, auto_save=False, **fmt)
            _queue_task("create_document_content")

        mark_doc_dirty_and_refresh()
        return f"SUCCESS: Initial document content was written.\n{result}"
    except Exception as e:
        return f"ERROR: Failed to create document content: {e}"


@tool
def add_line(
    paragraph: int, line: int, text: str,
    font_name: Optional[str] = _fmt_font_name(), font_size: Optional[float] = _fmt_font_size(),
    bold: Optional[bool] = _fmt_bold(), italic: Optional[bool] = _fmt_italic(),
    underline: Optional[bool] = _fmt_underline(), strikethrough: Optional[bool] = _fmt_strikethrough(),
    font_color: Optional[str] = _fmt_font_color(), highlight_color: Optional[str] = _fmt_highlight(),
    superscript: Optional[bool] = _fmt_superscript(), subscript: Optional[bool] = _fmt_subscript(),
    alignment: Optional[str] = _fmt_alignment(), style: Optional[str] = _fmt_style(),
) -> str:
    """Add a line to an existing document. For an empty document, use create_document_content.
    All formatting params (font_size, bold, italic, etc.) are optional — set any to apply formatting."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "ERROR: The document is empty. Use create_document_content for the initial content."

    fmt = _collect_fmt(**{k: v for k, v in locals().items() if k in _FMT_FIELDS})
    result = mgr.add_line(paragraph, line, text, auto_save=False, **fmt)
    _queue_task(f"add_line P{paragraph}L{line}")
    mark_doc_dirty_and_refresh()
    return result


@tool
def batch_edit_document(edits_json: str) -> str:
    """Apply multiple add/update/delete edits in one call.

    Each edit: {"action": "add"|"update"|"delete", "paragraph": int, "line": int, "text": "..."}
    Optional "format" object with the same fields as add_line/update_line formatting kwargs."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."

    try:
        edits = json.loads(edits_json)
    except json.JSONDecodeError as e:
        return f"ERROR: edits_json must be a valid JSON array. {e}"

    if not isinstance(edits, list) or not edits:
        return "ERROR: edits_json must be a non-empty JSON array."

    normalized = []
    for e in edits:
        if not isinstance(e, dict) or "action" not in e:
            return "ERROR: Every edit needs an 'action' key."
        action = str(e.get("action", "")).lower()
        if action not in {"add", "update", "delete"}:
            return f"ERROR: Unsupported action '{action}'. Use add, update, or delete."
        try:
            paragraph = int(e.get("paragraph"))
            line_num = int(e.get("line"))
        except (TypeError, ValueError):
            return "ERROR: paragraph and line must be integers."

        fmt = e.get("format") or {}
        if not isinstance(fmt, dict):
            fmt = {}
        normalized.append({
            "action": action, "paragraph": paragraph, "line": line_num,
            "text": e.get("text", ""), "format": fmt,
        })

    try:
        results = mgr.batch_edit(normalized, auto_save=False)
        _queue_task(f"batch_edit_document ({len(normalized)} edits)")
        mark_doc_dirty_and_refresh()
        return (
            f"Applied {len(results)} edit(s), queued for one save with the rest of this request:\n"
            + "\n".join(results)
        )
    except Exception as e:
        return f"ERROR: Batch edit failed: {e}"


# ═════════════════════════════════════════════════════════════════
#  UPDATER TOOLS
# ═════════════════════════════════════════════════════════════════

@tool
def update_line(
    paragraph: int, line: int, new_text: str,
    font_name: Optional[str] = _fmt_font_name(), font_size: Optional[float] = _fmt_font_size(),
    bold: Optional[bool] = _fmt_bold(), italic: Optional[bool] = _fmt_italic(),
    underline: Optional[bool] = _fmt_underline(), strikethrough: Optional[bool] = _fmt_strikethrough(),
    font_color: Optional[str] = _fmt_font_color(), highlight_color: Optional[str] = _fmt_highlight(),
    superscript: Optional[bool] = _fmt_superscript(), subscript: Optional[bool] = _fmt_subscript(),
    alignment: Optional[str] = _fmt_alignment(), style: Optional[str] = _fmt_style(),
) -> str:
    """Update an existing line's text and optionally its formatting.
    All formatting params (font_size, bold, italic, etc.) are optional — set any to apply formatting."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "ERROR: Cannot update an empty document because there is no existing content."

    fmt = _collect_fmt(**{k: v for k, v in locals().items() if k in _FMT_FIELDS})
    result = mgr.update_line(paragraph, line, new_text, auto_save=False, **fmt)
    _queue_task(f"update_line P{paragraph}L{line}")
    mark_doc_dirty_and_refresh()
    return result


@tool
def format_line(
    paragraph: int, line: int,
    font_name: Optional[str] = _fmt_font_name(), font_size: Optional[float] = _fmt_font_size(),
    bold: Optional[bool] = _fmt_bold(), italic: Optional[bool] = _fmt_italic(),
    underline: Optional[bool] = _fmt_underline(), strikethrough: Optional[bool] = _fmt_strikethrough(),
    font_color: Optional[str] = _fmt_font_color(), highlight_color: Optional[str] = _fmt_highlight(),
    superscript: Optional[bool] = _fmt_superscript(), subscript: Optional[bool] = _fmt_subscript(),
    alignment: Optional[str] = _fmt_alignment(), style: Optional[str] = _fmt_style(),
) -> str:
    """Change a line's formatting WITHOUT touching its text. Use for font size, bold, italic, color, alignment changes.
    Example: format_line(paragraph=1, line=1, bold=True, font_size=10.0)"""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "ERROR: Cannot format content because the document is empty."

    fmt = _collect_fmt(**{k: v for k, v in locals().items() if k in _FMT_FIELDS})
    if not fmt:
        return "ERROR: No formatting fields given -- nothing to change."

    result = mgr.format_line(paragraph, line, auto_save=False, **fmt)
    _queue_task(f"format_line P{paragraph}L{line}")
    mark_doc_dirty_and_refresh()
    return result


@tool
def replace_all(old_text: str, new_text: str) -> str:
    """Replace text across the entire document."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "ERROR: Cannot replace content because the document is empty."

    result = mgr.replace_all(old_text, new_text, auto_save=False)
    _queue_task("replace_all")
    mark_doc_dirty_and_refresh()
    return result


# ═════════════════════════════════════════════════════════════════
#  DELETER TOOL
# ═════════════════════════════════════════════════════════════════

@tool
def delete_line(paragraph: int, line: int) -> str:
    """Delete a specific line."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    if _document_is_empty():
        return "ERROR: Cannot delete content because the document is empty."

    result = mgr.delete_line(paragraph, line, auto_save=False)
    _queue_task(f"delete_line P{paragraph}L{line}")
    mark_doc_dirty_and_refresh()
    return result


# ═════════════════════════════════════════════════════════════════
#  SHARED TOOLS
# ═════════════════════════════════════════════════════════════════

@tool
def show_history(check: str = "all") -> str:
    """Show document change history."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    return mgr.history()


@tool
def save_copy(output_path: str) -> str:
    """Save the current document to a new file."""
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."
    return mgr.save_as(output_path)


@tool
def reload_from_disk(confirm: str = "yes") -> str:
    """Reload document from disk."""
    if confirm.lower() != "yes":
        return "Reload cancelled. Pass confirm='yes' to proceed."
    mgr = _state.get("doc_manager")
    if not mgr:
        return "ERROR: No document loaded."

    _state.get("task_queue", deque()).clear()
    result = mgr.reload()
    _state["doc_data"] = mgr.doc_data
    _state["_dirty"] = False
    return result


# ═════════════════════════════════════════════════════════════════
#  PROMPTS & TOOL SETS
# ═════════════════════════════════════════════════════════════════

_DOC_DATA_MODEL = """

DOCUMENT DATA MODEL:
- The document is a flat list of lines.  Each line is addressed by paragraph (P) and line (L).
- P numbers come from the Word document's real paragraph order and CAN skip numbers (blank paragraphs for spacing).
- Never invent an existing P/L address.
- For an EXISTING document, search first to determine the correct P/L address.
- L numbers are 1-indexed.
- Adding a line shifts later lines in the same paragraph.
- If multiple lines are added to the same paragraph, use batch_edit_document in correct insertion order.
- If the document is completely empty, there is no existing P/L address to search for.

FORMATTING AND HEADINGS:
- You CAN change font size, bold, italic, underline, strikethrough, font color, highlight color, superscript, subscript, alignment, and paragraph style.
- Use format_line to change formatting WITHOUT touching the text.  Use update_line to change BOTH text and formatting.
- Available formatting kwargs on add_line / update_line / format_line:
    font_name (str), font_size (float, in pt), bold (bool), italic (bool),
    underline (bool), strikethrough (bool), font_color (hex RGB string, no '#'),
    highlight_color (one of: "yellow", "bright_green", "turquoise", "pink", "blue",
    "red", "dark_blue", "teal", "green", "violet", "dark_red", "dark_yellow",
    "gray_50", "gray_25", "black", "white", "none"),
    superscript (bool), subscript (bool), alignment ("left"/"center"/"right"/"justify"),
    style (paragraph style, e.g. "Heading 1".."Heading 9", "Title", "Normal").
- A paragraph becomes/stops being a heading only via the `style` kwarg -- heading_level is derived from style.
- Use get_document_outline to see the heading structure before deciding where new content belongs.
- Use get_paragraph_style to check a paragraph's current style before changing it.
"""

_DATA_SUFFICIENCY_POLICY = """

DATA SUFFICIENCY AND WRITE POLICY:
IMPORTANT DISTINCTION:
1. Inventing or pretending that information already exists in the document is PROHIBITED.
2. Generating NEW information that the user explicitly asked you to add is ALLOWED and REQUIRED.

EMPTY DOCUMENT:
- An empty document does NOT prevent an explicit ADD/CREATE/WRITE/INSERT/POPULATE/FILL request.
- Generate appropriate content for exactly what the user requested and actually write it.
- Do NOT repeatedly search an empty document or ask the user to provide content again.
- For an empty document, use create_document_content.

EMPTY DOCUMENT + READ: Explain that no information exists.
EMPTY DOCUMENT + UPDATE: Explain that there is no existing content to update.
EMPTY DOCUMENT + DELETE: Explain that there is nothing to delete.

NON-EMPTY DOCUMENT:
- Do not invent existing document content.
- For update/delete, search for the actual content first.
- An explicit ADD request may still be fulfilled even with little existing content.

WRITE REQUIREMENT:
- Never claim a document was modified unless an actual write tool was called successfully.
- Writing text in the final response does NOT modify the file.
"""

_READER_PROMPT = (
    "ROLE: Document Reader\n"
    "GOAL: Search and understand existing document content with precision.\n\n"
    "Use search_document for exact text, vector_search for semantic meaning, "
    "find_full_sentence when complete sentences are needed, find_with_context when "
    "surrounding context is needed.\n\n"
    "If the document is empty, do not pretend that content exists."
    + _DOC_DATA_MODEL + _DATA_SUFFICIENCY_POLICY
)

_READER_TOOLS = [
    search_document, vector_search, find_full_sentence, find_with_context,
    get_paragraph_style, get_document_outline,
]

_WRITER_PROMPT = (
    "ROLE: Document Writer\n"
    "GOAL: Create and add new content to the document, with optional formatting.\n\n"
    "CRITICAL: If the user asks you to add/create/write/insert/populate/fill/generate/compose/append, "
    "you MUST perform the actual write operation.\n\n"
    "EMPTY DOCUMENT: Use create_document_content.  Do NOT search first.\n"
    "NON-EMPTY DOCUMENT: Search to determine where new content belongs.\n"
    "MULTI-LINE: Prefer batch_edit_document for large additions.\n\n"
    "*** FORMATTING — YOU HAVE THESE TOOLS: ***\n"
    "- add_line(paragraph, line, text='...', bold=True, font_size=14.0) → adds formatted text\n"
    "- create_document_content(text='...', bold=True, font_size=12.0)   → creates with formatting\n"
    "- format_line(paragraph, line, bold=True, font_size=10.0)          → changes existing formatting\n"
    "NEVER say you cannot change formatting. You have full control.\n\n"
    "You are NOT finished until the appropriate write tool has actually been called."
    + _DOC_DATA_MODEL + _DATA_SUFFICIENCY_POLICY
)

_WRITER_TOOLS = [
    search_document, vector_search, add_line, format_line,
    batch_edit_document, create_document_content, show_history,
    get_paragraph_style, get_document_outline,
]

_UPDATER_PROMPT = (
    "ROLE: Document Updater\n"
    "GOAL: Modify existing document content and formatting accurately.\n\n"
    "Workflow: 1) Search for content  2) Identify exact P/L  3) Perform update  4) Verify.\n"
    "If the document is empty, there is no existing content to update.\n"
    "Use replace_all for global changes, batch_edit_document for multiple changes.\n\n"
    "*** FORMATTING — YOU HAVE THESE TOOLS: ***\n"
    "- format_line(paragraph, line, font_size=10.0)  → changes font size to 10pt\n"
    "- format_line(paragraph, line, bold=True)        → makes text bold\n"
    "- format_line(paragraph, line, italic=True)      → makes text italic\n"
    "- format_line(paragraph, line, font_color='FF0000') → red text\n"
    "- format_line(paragraph, line, style='Heading 1')   → changes paragraph style\n"
    "- update_line(paragraph, line, new_text='...', bold=True, font_size=14.0) → changes text AND formatting\n\n"
    "You MUST use format_line or update_line for ANY formatting request. "
    "NEVER say you cannot change font size, bold, italic, colors, or any formatting. "
    "You have full formatting control through these tools.\n\n"
    "An actual update/format tool must be called — the final response is not the modification."
    + _DOC_DATA_MODEL + _DATA_SUFFICIENCY_POLICY
)

_UPDATER_TOOLS = [
    search_document, vector_search, update_line, format_line,
    replace_all, batch_edit_document, show_history,
    get_paragraph_style, get_document_outline,
]

_DELETER_PROMPT = (
    "ROLE: Document Deleter\n"
    "GOAL: Safely remove existing document content.\n\n"
    "Workflow: 1) Search for content  2) Confirm exact P/L  3) Delete  4) Verify.\n"
    "If the document is empty, there is nothing to delete.\n"
    "Use batch_edit_document for multiple deletions.\n"
    "A delete tool must actually be called — the final answer does not delete anything."
    + _DOC_DATA_MODEL + _DATA_SUFFICIENCY_POLICY
)

_DELETER_TOOLS = [search_document, vector_search, delete_line, batch_edit_document, show_history]

_SPECIALIST_DEFS = {
    "reader":  (_READER_PROMPT,  _READER_TOOLS),
    "writer":  (_WRITER_PROMPT,  _WRITER_TOOLS),
    "updater": (_UPDATER_PROMPT, _UPDATER_TOOLS),
    "deleter": (_DELETER_PROMPT, _DELETER_TOOLS),
}


# ═════════════════════════════════════════════════════════════════
#  SPECIALIST SUBGRAPHS
# ═════════════════════════════════════════════════════════════════

class SpecialistState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_specialist_subgraph(node_name: str, system_prompt: str, tools: list, llm):
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: SpecialistState) -> SpecialistState:
        response = llm_with_tools.invoke(
            [SystemMessage(content=system_prompt)] + state["messages"]
        )
        return {"messages": [response]}

    graph = StateGraph(SpecialistState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "agent")
    return graph.compile(name=node_name)


def build_specialists(llm) -> Dict:
    return {
        name: _build_specialist_subgraph(name, prompt, tools, llm)
        for name, (prompt, tools) in _SPECIALIST_DEFS.items()
    }


# ═════════════════════════════════════════════════════════════════
#  ROUTER (keyword-based, no LLM call)
# ═════════════════════════════════════════════════════════════════

_INTENT_ROUTES = [
    # Delete first: "delete old pricing and add new" should remove before adding
    ("deleter", ("delete", "remove", "erase")),
    ("updater", (
        "update", "edit", "modify", "change", "replace", "fix", "correct",
        "rewrite", "rename", "refactor", "space", "spacing", "blank", "gap",
        "format", "formatting", "indent", "align", "line break", "empty line",
        "bold", "italic", "underline", "font", "size", "color", "highlight",
        "heading", "title", "style", "strikethrough", "superscript", "subscript",
    )),
    ("writer", (
        "add", "insert", "append", "continue", "write", "create", "populate",
        "fill", "generate", "compose", "draft", "include", "put", "introduce",
    )),
]


def route_agents(user_request: str) -> List[str]:
    """Determine which specialists should run (no LLM router)."""
    text = user_request.lower()
    matched = [
        name for name, keywords in _INTENT_ROUTES
        if any(_contains_keyword(text, kw) for kw in keywords)
    ]
    if matched:
        return matched
    if _looks_like_write_request(user_request):
        return ["updater"]
    return ["reader"]


# ═════════════════════════════════════════════════════════════════
#  CREW GRAPH
# ═════════════════════════════════════════════════════════════════

class CrewState(TypedDict):
    messages: Annotated[list, add_messages]
    pending: List[str]


def _route_node(state: CrewState) -> CrewState:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)), None
    )
    text = last_human.content if last_human else ""
    if not isinstance(text, str):
        text = str(text)
    return {"pending": route_agents(text)}


def _advance_node(state: CrewState) -> CrewState:
    return {"pending": state["pending"][1:]}


def _dispatch(state: CrewState) -> str:
    pending = state.get("pending") or []
    return pending[0] if pending else END


def build_crew_graph(specialists: Dict):
    graph = StateGraph(CrewState)
    graph.add_node("route", _route_node)
    graph.add_node("advance", _advance_node)
    for name, subgraph in specialists.items():
        graph.add_node(name, subgraph)

    dispatch_map = {name: name for name in specialists}
    dispatch_map[END] = END

    graph.add_edge(START, "route")
    graph.add_conditional_edges("route", _dispatch, dispatch_map)
    for name in specialists:
        graph.add_edge(name, "advance")
    graph.add_conditional_edges("advance", _dispatch, dispatch_map)
    return graph.compile()


# ═════════════════════════════════════════════════════════════════
#  RESPONSE HELPERS
# ═════════════════════════════════════════════════════════════════

def _extract_text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(part["text"])
            elif isinstance(part, str):
                parts.append(part)
        return "\n".join(parts)
    return str(content)


def _last_answer_text(messages: list) -> str:
    """Return the last AI message containing visible text."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            text = _extract_text(m.content).strip()
            if text:
                return text
    return "(no response)"


# ═════════════════════════════════════════════════════════════════
#  MAIN SYSTEM
# ═════════════════════════════════════════════════════════════════

class DocsAgentSystem:

    def __init__(self, filepath: str, llm=None):
        self.filepath = filepath

        # LLM
        if llm:
            self.llm = llm
        else:
            from doxl_ai_terminal.pipeline.config import get_agentic_llm
            from doxl_ai_terminal.pipeline.credential_store import load_credentials
            cred = load_credentials()
            if not cred:
                raise RuntimeError("No saved Gemini credentials found. Run onboarding first or pass llm=.")
            info(f"Using saved credentials (model: {cred['model']})")
            self.llm = get_agentic_llm(cred["api_key"], cred["model"])

        # Load document
        header("DOXL AI — Document Agent")
        process_file(filepath)

        # Build graphs once
        self.specialists = build_specialists(self.llm)
        self.graph = build_crew_graph(self.specialists)

        # UI
        header("Ready", "Reader · Writer · Updater · Deleter")
        info(
            "Ask in plain English, e.g. \"find mentions of revenue\".\n"
            "  'history' shows changes made, 'quit' exits.\n"
            "  Full-document 'view' is disabled — reports can run 100+ pages."
        )

    # ── Request processing ───────────────────────────────────────

    def process_request(self, user_input: str) -> str:
        mgr = _state.get("doc_manager")
        changes_before = len(mgr.changelog.changes) if mgr else 0

        # Execute graph
        result_state = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "pending": []},
            config={"recursion_limit": 150},
        )

        # Drain edit queue and save once
        flush_task_queue()
        rebuild_doc_vector_index_if_dirty()

        # Ground truth verification
        mgr = _state.get("doc_manager")
        changes_after = len(mgr.changelog.changes) if mgr else 0
        made = changes_after - changes_before

        result_str = _last_answer_text(result_state["messages"])

        if made > 0:
            result_str += f"\n\n[VERIFIED] {made} change(s) were actually written to the file."
        elif _looks_like_write_request(user_input):
            result_str += (
                "\n\n[WARNING] This request appears to require a document modification, "
                "but no write tool was actually called. The file was NOT modified."
            )
        return result_str

    # ── Chat loop ────────────────────────────────────────────────

    def chat(self):
        while True:
            try:
                user_input = prompt_input("\nYou: ")
            except (KeyboardInterrupt, EOFError):
                print()
                success("Goodbye!")
                break

            if not user_input:
                continue

            normalized = user_input.strip().lower()

            if normalized in ("quit", "exit", "q", "bye"):
                success("Goodbye!")
                break

            if normalized == "view":
                warn(
                    "'view' is disabled — a document can run 100+ pages. "
                    "Try searching naturally instead, e.g. 'find the section about pricing'."
                )
                continue

            if normalized == "history":
                console_print = info
                console_print(show_history.invoke("all"))
                continue

            spinner = Spinner("Thinking...").start()
            try:
                # Stop spinner before graph runs so TaskProgress counter
                # doesn't conflict with Rich's one-live-display limit.
                spinner.stop(ok=False)
                result = self.process_request(user_input)
                agent_result(result)
            except Exception as e:
                spinner.stop(ok=False)
                global _task_progress
                if _task_progress:
                    _task_progress.finish(saved=False)
                    _task_progress = None
                error(str(e) if str(e) != "None" else "Request failed — the agent could not complete the task.")
                info("Try rephrasing your request.")


# ═════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        info("Usage: python docs_agent.py <filepath.docx>")
        info("Example: python docs_agent.py report.docx")
        sys.exit(1)

    filepath = sys.argv[1]
    system = DocsAgentSystem(filepath, llm=None)
    system.chat()


if __name__ == "__main__":
    main()
