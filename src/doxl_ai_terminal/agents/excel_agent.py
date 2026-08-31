# excel_agent.py
"""
LangGraph Multi-Agent Excel Handler
==================================

Architecture:
    Code Router (route_agents — keyword-based, no LLM call)
        ├── Reader Subgraph        → list sheets, exact search, RAG semantic search
        ├── Writer Subgraph        → add cells, add rows
        ├── Updater Subgraph       → update cells, find-replace
        ├── Deleter Subgraph       → delete cells, delete rows
        └── Formula Gen Subgraph   → inspect data, craft & insert Excel formulas

    Each specialist above is its own compiled LangGraph StateGraph
    (a ReAct-style agent ↔ tools loop) added as a SUBGRAPH node into
    one parent "crew" graph (see build_crew_graph). All subgraphs
    share the same `messages` key (via LangGraph's add_messages
    reducer), so when more than one specialist runs for a request,
    the second one sees the first one's tool calls/results as
    conversation history automatically.

    NOTE: this used to be CrewAI, first with a Manager Agent
    (Process.hierarchical) that delegated at runtime via its own LLM
    reasoning — costing an extra "who should handle this?" call plus
    a review-the-result call on EVERY request — then briefly a
    CrewAI Process.sequential crew with the same manager removed.
    It's now rebuilt on LangGraph: the same keyword router picks the
    specialist(s) in plain Python (route_agents, below), and each
    specialist runs as its own compiled subgraph inside one parent
    graph, instead of an LLM manager or a third-party crew runtime.
    See build_crew_graph's docstring.

Flow:
    1. User gives filepath → file is read, chunked, vectorized
    2. Chatbot loop starts
    3. User types a request
    4. route() node picks the specialist(s) by keyword — no LLM call
    5. Each specialist subgraph runs in turn (agent ↔ tools loop),
       sharing one growing `messages` list → result returned
    6. Loop continues until user quits

Usage:
    python excel_agent.py sales_data.xlsx
"""

import os
import sys
import json
from typing import Optional, List, Dict, TypedDict, Annotated

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

# ---- Your existing modules ----
from doxl_ai_terminal.Frontier.changelogmanager_excel import LiveExcelManager
from doxl_ai_terminal.Frontier.fileReader import read_excel
from doxl_ai_terminal.Chunker.chunking import chunk_excel_row_wise, chunk_excel_column_wise
from doxl_ai_terminal.data_handler.vector_config import store_in_chroma, make_collection_name
from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager
from doxl_ai_terminal.pipeline.config import SUPPORTED_MODELS, validate_api_key, get_agentic_llm
from doxl_ai_terminal.pipeline.credential_store import (
    has_credentials,
    load_credentials,
    save_credentials,
)
from doxl_ai_terminal.pipeline.terminal_ui import (
    Spinner, info, success, warn, error, header, agent_result, prompt_input,
)
# ================================E ================================
# SECTION 1: SHARED STATE
# ================================================================
# All tools read from this dict to access the loaded spreadsheet.
# Same pattern as manager_tools.py and docs_agent.py

_state: Dict = {
    "excel_manager": None,      # LiveExcelManager  (edit + auto-save)
    "excel_data": None,         # Raw ExcelData      (for reference)
    "vector_managers": {},      # {format_name: VectorDBManager} — one per chunking strategy
    "collections": [],          # Collection names created
    "filepath": None,           # Original file path
    "_dirty": False,            # True when a write happened but the vector index hasn't caught up yet
}


# ================================================================
# SECTION 2: FILE PROCESSOR
# ================================================================
# Mimics pipeliner.py — read → chunk → vector store
# PLUS loads LiveExcelManager for live editing

def process_file(filepath: str):
    """
    Full pipeline: Read the .xlsx → Chunk it → Store in vector DB → Load editor.

    After this, all tools can operate on the loaded spreadsheet.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".xlsx":
        raise ValueError(f"Only .xlsx files supported. Got: {ext}")

    _state["filepath"] = filepath

    # The read/chunk/embed/index steps below are all "behind the scenes"
    # plumbing the user doesn't need to watch line-by-line — one spinner
    # covers the whole load instead of four separate step prints.
    with Spinner("Loading the file..."):
        # ---- Step 1: Read the spreadsheet into ExcelData ----
        excel_data = read_excel(filepath)
        _state["excel_data"] = excel_data

        sheet_names = [s.sheet_name for s in excel_data.sheets]
        total_cells = sum(len(s.cells) for s in excel_data.sheets)

        # ---- Step 2: Load LiveExcelManager for editing ----
        _state["excel_manager"] = LiveExcelManager(filepath)

        # ---- Steps 3-4: Chunk + store in vector DB + load for search ----
        _rebuild_excel_vector_index(filepath, excel_data)

    info(f"{len(sheet_names)} sheet(s): {', '.join(sheet_names)}")
    success(f"Ready! {total_cells} cells across {len(sheet_names)} sheet(s).\n")


# ================================================================
# SECTION 2B: RE-INDEX — keep the vector store in sync with writes
# ================================================================

CHUNK_STRATEGIES = [
    ("row_wise", chunk_excel_row_wise),
    ("column_wise", chunk_excel_column_wise),
]


def _rebuild_excel_vector_index(filepath: str, excel_data) -> None:
    """
    (Re)build every chunking strategy's Chroma collection for this
    file and load ALL of them into _state["vector_managers"], keyed
    by format name (row_wise / column_wise) — not just one default.
    Previously only row_wise was ever loaded, so every column_wise
    chunk got embedded and then never searched.

    The collection name is derived from the file's full path (see
    make_collection_name), so two different files that happen to
    share a filename never collide. Any existing collection for this
    exact file+format is deleted before rebuilding, so re-embedding
    (a fresh open OR a post-write reindex) replaces stale chunks
    instead of piling duplicates on top of them.
    """
    db = VectorDBManager()
    collections = []
    vector_managers = {}

    for format_name, chunk_fn in CHUNK_STRATEGIES:
        chunks = chunk_fn(excel_data)
        if not chunks:
            continue

        collection_name = make_collection_name(filepath, format_name)
        try:
            db.delete_collection(collection_name)
        except Exception:
            pass  # nothing to delete yet, e.g. first time this file is opened

        store_in_chroma(chunks, collection_name)
        collections.append(collection_name)
        vector_managers[format_name] = VectorDBManager().load_collection(collection_name)

    _state["collections"] = collections
    _state["vector_managers"] = vector_managers


def _refresh_excel_data() -> None:
    """
    Re-read the (now auto-saved) file from disk into _state["excel_data"].

    This is the CHEAP half of keeping things in sync — just re-parsing
    the .xlsx with openpyxl — so the formula tools (get_data_bounds,
    get_column_data, get_row_data, insert_formula's existing-cell
    check) always see the current sheet, even mid-request, before the
    (much more expensive) vector re-embed below has run.
    """
    filepath = _state.get("filepath")
    if not filepath:
        return
    _state["excel_data"] = read_excel(filepath)
    _state["_dirty"] = True  # vector index now needs rebuilding


def mark_excel_dirty_and_refresh() -> None:
    """Called by every mutating tool right after it changes the sheet."""
    try:
        _refresh_excel_data()
    except Exception as e:
        warn(f"Refreshing sheet data failed: {e}")


def rebuild_excel_vector_index_if_dirty() -> None:
    """
    Rebuild the (expensive) Chroma vector index, but ONLY if something
    actually changed since the last rebuild — and only once per user
    request rather than once per tool call.

    Chunking + re-embedding every collection is the slow part of a
    write (re-chunk the whole sheet, run it through sentence-transformers,
    rewrite Chroma). Doing that after every single add_cell/update_cell/
    delete_row call — which is what earlier versions did — meant a
    compound request like "find all rows under 100 and delete them"
    paid that cost once per row deleted. Calling this once, right
    after the crew finishes the whole request, keeps vector_search_excel
    accurate for the user's NEXT message while paying the expensive
    part exactly once per turn instead of once per edit.
    """
    if not _state.get("_dirty"):
        return
    filepath = _state.get("filepath")
    excel_data = _state.get("excel_data")
    if not filepath or excel_data is None:
        return
    try:
        _rebuild_excel_vector_index(filepath, excel_data)
        _state["_dirty"] = False
    except Exception as e:
        warn(f"Vector index refresh failed: {e}")


# Rough keyword heuristic used only to decide whether to show a
# "no write tool was called" warning after a request — see
# ExcelAgentSystem.process_request(). False positives just mean an
# occasional unnecessary warning; false negatives mean a missed one,
# which is the failure mode this exists to catch, so the list is kept
# broad on purpose. Also doubles as the write-intent signal route_agents
# falls back on below when nothing in _INTENT_ROUTES matched — see the
# matching comment in docs_agent.py for the concrete case (a formatting
# request with none of update/add/delete's trigger words) this exists to
# catch.
_WRITE_INTENT_KEYWORDS = (
    "add", "insert", "append", "continue", "update", "edit", "modify",
    "change", "replace", "delete", "remove", "fix", "correct", "fill",
    "set ", "rename", "formula", "calculate", "sum", "vlookup",
    "space", "spacing", "blank", "gap", "format", "formatting",
    "indent", "align", "empty row", "empty column",
)


def _looks_like_write_request(user_input: str) -> bool:
    text = user_input.lower()
    return any(kw in text for kw in _WRITE_INTENT_KEYWORDS)


# ================================================================
# SECTION 3: TOOLS — READER
# ================================================================
# These tools let the Reader specialist search the spreadsheet. There
# is deliberately NO "view everything" tool here — see the note below.
# Each tool is a LangChain @tool (langchain_core.tools.tool). LangGraph's
# ToolNode expects plain LangChain BaseTool instances, so every tool
# here is bound via llm.bind_tools([...]) and executed by a ToolNode
# inside each specialist's subgraph (see SECTION 8B).

# ─────────────────────────────────────────────────────────────────
# DISABLED: view_spreadsheet used to dump every cell in the workbook
# into one giant string. For a sheet with lakhs of rows (or even a
# modest multi-sheet workbook), that's hundreds of thousands of
# tokens handed to the LLM in one shot — it blows the context window
# and is painfully slow. Use search_spreadsheet (exact keyword) or
# vector_search_excel (RAG semantic search) instead: both return only
# the handful of matching cells/chunks that are actually relevant, so
# they scale to a file of any size.
#
# @tool
# def view_spreadsheet(sheet_name: str = "") -> str:
#     """View all cells in the loaded Excel file.
#     Pass a sheet name to filter by one sheet, or leave empty to see all.
#     Shows sheet name, cell position [ColumnRow], and value."""
#     mgr = _state["excel_manager"]
#     if not mgr:
#         return "ERROR: No spreadsheet loaded."
#     return mgr.view(sheet_name if sheet_name else None)
# ─────────────────────────────────────────────────────────────────


@tool
def search_spreadsheet(keyword: str) -> str:
    """Search for an exact keyword across all cells in the spreadsheet.
    Returns every cell containing that keyword with sheet name and position.
    Use this for exact text matching."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    results = mgr.search(keyword)
    if not results:
        return f"No matches found for '{keyword}'."
    lines = [f"{r['sheet']}[{r['cell']}] = {r['value']}" for r in results]
    return "\n".join(lines)


# Chroma's default distance space is squared L2. Because embeddings are
# normalized (unit vectors), L2^2 = 2 - 2*cosine_similarity, so ranking
# by L2^2 ascending is equivalent to ranking by cosine similarity
# descending. This is that L2^2 cutoff (~ cosine similarity >= 0.35) —
# a reasonable starting default so a query with no real match in the
# file returns "no matches" instead of forcing back 4 unrelated chunks.
# Tune it if results feel too strict/loose for your data.
MAX_RELEVANT_DISTANCE = 1.3


@tool
def vector_search_excel(query: str) -> str:
    """Semantic search — find content by MEANING, not just exact words.
    Searches across every chunking strategy for this file (row-wise AND
    column-wise) and merges the results, so nothing is missed just
    because it only surfaces well in one view of the data.
    Example: searching 'profit' also finds 'revenue', 'margin', 'earnings'.
    Returns the top matches from the spreadsheet — or says so plainly
    if nothing in the file is actually relevant."""
    managers = _state.get("vector_managers") or {}
    if not managers:
        return "ERROR: Vector store not loaded."

    candidates = []  # list of (Document, distance)
    for format_name, vm in managers.items():
        try:
            candidates.extend(vm.search_with_scores(query, k=4))
        except Exception as e:
            return f"Search error ({format_name}): {e}"

    # Keep only genuinely relevant hits, then take the best 4 overall
    # across BOTH chunking strategies (lower distance = more similar).
    relevant = [pair for pair in candidates if pair[1] <= MAX_RELEVANT_DISTANCE]
    relevant.sort(key=lambda pair: pair[1])
    top = relevant[:4]

    if not top:
        return f"No semantic matches for '{query}'."

    output = []
    for i, (doc, score) in enumerate(top, 1):
        sheet = doc.metadata.get("sheet", "?")
        fmt = doc.metadata.get("format", "?")
        if fmt == "column_wise":
            where = f"rows {doc.metadata.get('row_start', '?')}-{doc.metadata.get('row_end', '?')}"
        else:
            where = f"row {doc.metadata.get('row', '?')}"
        output.append(f"[{i}] ({sheet}, {where}, {fmt}) {doc.page_content}")
    return "\n".join(output)


@tool
def list_sheets(check: str = "all") -> str:
    """List all sheet names in the spreadsheet with their cell counts.
    Useful to know which sheets exist before doing any operation."""
    excel_data = _state["excel_data"]
    if not excel_data:
        return "ERROR: No spreadsheet loaded."
    lines = []
    for sheet in excel_data.sheets:
        lines.append(f"  {sheet.sheet_name}: {len(sheet.cells)} cells, "
                      f"{sheet.total_rows} rows x {sheet.total_columns} columns")
    return f"Sheets in '{excel_data.filename}':\n" + "\n".join(lines)


# ================================================================
# SECTION 4: TOOLS — WRITER
# ================================================================

@tool
def add_cell(sheet_name: str, row: str, column: str, value: str) -> str:
    """Add a new cell to the spreadsheet. If the target position already
    holds a value, it is UPDATED in place instead of creating a duplicate.
    Args:
        sheet_name: name of the sheet (use list_sheets to see names)
        row: row number as string (e.g., '5')
        column: column letter (e.g., 'A', 'B', 'C')
        value: the text/number to put in the cell
    Auto-saves to file after adding."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    result = mgr.add_cell(sheet_name, row, column, value)
    mark_excel_dirty_and_refresh()
    return result


@tool
def add_row(sheet_name: str, row_data_str: str) -> str:
    """Add a full row of data to the spreadsheet.
    The row is appended after the last existing row.
    Args:
        sheet_name: name of the sheet
        row_data_str: column-value pairs as 'A=value1,B=value2,C=value3'
    Example: add_row('Sheet1', 'A=John,B=25,C=Chennai')
    Auto-saves to file after adding."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."

    # Parse the string into a dict: "A=val1,B=val2" -> {"A": "val1", "B": "val2"}
    row_data = {}
    try:
        for pair in row_data_str.split(","):
            col, val = pair.strip().split("=", 1)
            row_data[col.strip()] = val.strip()
    except ValueError:
        return "ERROR: Format must be 'A=value1,B=value2,C=value3'"

    result = mgr.add_row(sheet_name, row_data)
    mark_excel_dirty_and_refresh()
    return result


# ================================================================
# SECTION 5: TOOLS — UPDATER
# ================================================================

@tool
def update_cell(sheet_name: str, row: str, column: str, new_value: str) -> str:
    """Update an existing cell in the spreadsheet.
    Args:
        sheet_name: name of the sheet
        row: row number as string (e.g., '3')
        column: column letter (e.g., 'B')
        new_value: the new value to put in the cell
    IMPORTANT: Use search_spreadsheet first to find the exact position.
    Auto-saves to file after updating."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    result = mgr.update_cell(sheet_name, row, column, new_value)
    mark_excel_dirty_and_refresh()
    return result


@tool
def replace_all_excel(old_text: str, new_text: str) -> str:
    """Find and replace text across ALL cells in ALL sheets.
    Args:
        old_text: the text to find
        new_text: the text to replace it with
    Every occurrence of old_text will be replaced. Auto-saves to file."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    result = mgr.replace_all(old_text, new_text)
    mark_excel_dirty_and_refresh()
    return result


@tool
def batch_edit_excel(edits_json: str) -> str:
    """Apply MULTIPLE add/update/delete edits in ONE call, with a single
    file save at the end — use this instead of calling add_cell /
    update_cell / delete_cell repeatedly whenever a request touches more
    than a couple of cells (e.g. "add 20 dummy rows", "update every row
    where status is Pending"). This exists because relying on the model
    to make N separate tool calls in a row is unreliable — on bulk
    requests it has been observed to just DESCRIBE all N changes in its
    final answer instead of actually calling the tool N times, which
    means nothing gets saved even though the response claims success.
    One call to this tool with all N edits removes that failure mode.

    Args:
        edits_json: a JSON array of edit objects, e.g.:
          [
            {"action": "add", "sheet": "Sheet1", "row": "6", "column": "A", "value": "Dummy 6"},
            {"action": "add", "sheet": "Sheet1", "row": "7", "column": "A", "value": "Dummy 7"},
            {"action": "update", "sheet": "Sheet1", "row": "3", "column": "B", "value": "New"},
            {"action": "delete", "sheet": "Sheet1", "row": "9", "column": "C"}
          ]
        "value" is required for add/update and ignored for delete.
        An "add" onto a cell that already has a value updates it in
        place rather than creating a duplicate, so it's safe to use
        even if you're not certain a cell is empty.
    Auto-saves to file ONCE after all edits are applied."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."

    try:
        edits = json.loads(edits_json)
    except json.JSONDecodeError as e:
        return f"ERROR: edits_json must be a valid JSON array of edit objects. {e}"

    if not isinstance(edits, list) or not edits:
        return "ERROR: edits_json must be a non-empty JSON array of edit objects."

    normalized = []
    for e in edits:
        if not isinstance(e, dict) or "action" not in e:
            return f"ERROR: every edit needs an 'action' key. Bad edit: {e}"
        normalized.append({
            "action": e.get("action"),
            "sheet": e.get("sheet"),
            "row": str(e.get("row", "")),
            "column": e.get("column"),
            "value": e.get("value", ""),
        })

    results = mgr.batch_edit(normalized)
    mark_excel_dirty_and_refresh()
    return f"Applied {len(results)} edit(s) in one save:\n" + "\n".join(results)


# ================================================================
# SECTION 6: TOOLS — DELETER
# ================================================================

@tool
def delete_cell(sheet_name: str, row: str, column: str) -> str:
    """Delete a specific cell from the spreadsheet.
    Args:
        sheet_name: name of the sheet
        row: row number as string (e.g., '3')
        column: column letter (e.g., 'B')
    IMPORTANT: Use search_spreadsheet first to confirm the exact cell.
    Auto-saves to file after deleting."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    result = mgr.delete_cell(sheet_name, row, column)
    mark_excel_dirty_and_refresh()
    return result


@tool
def delete_row(sheet_name: str, row: str) -> str:
    """Delete an entire row (all cells in that row) from a sheet.
    Args:
        sheet_name: name of the sheet
        row: row number as string (e.g., '5')
    WARNING: This removes ALL cells in the row. Use search_spreadsheet or
    vector_search_excel first to confirm the right row.
    Auto-saves to file after deleting."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    result = mgr.delete_row(sheet_name, row)
    mark_excel_dirty_and_refresh()
    return result


# ================================================================
# SECTION 6B: TOOLS — FORMULA GENERATOR
# ================================================================
# These tools let the Formula specialist inspect data and insert formulas.

@tool
def insert_formula(sheet_name: str, row: str, column: str, formula: str) -> str:
    """Insert an Excel formula into a specific cell.
    Args:
        sheet_name: name of the sheet
        row: row number as string (e.g., '10')
        column: column letter (e.g., 'C')
        formula: the Excel formula starting with '=' (e.g., '=SUM(B2:B9)')
    The formula is stored as-is and will be evaluated when opened in Excel.
    Auto-saves to file after inserting."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."

    # Make sure formula starts with =
    if not formula.startswith("="):
        formula = "=" + formula

    # Check if cell already exists → update it. Otherwise → add it.
    excel_data = _state["excel_data"]
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            for cell in sheet.cells:
                if cell.row == row and cell.column == column:
                    # Cell exists — update
                    result = mgr.update_cell(sheet_name, row, column, formula)
                    mark_excel_dirty_and_refresh()
                    return f"Formula inserted (updated): {sheet_name}[{column}{row}] = {formula}\n{result}"

    # Cell doesn't exist — add
    result = mgr.add_cell(sheet_name, row, column, formula)
    mark_excel_dirty_and_refresh()
    return f"Formula inserted (new cell): {sheet_name}[{column}{row}] = {formula}\n{result}"


@tool
def get_column_data(sheet_name: str, column: str) -> str:
    """Get all values in a specific column with row numbers.
    Essential before writing a formula — you need to know the exact range.
    Args:
        sheet_name: name of the sheet
        column: column letter (e.g., 'B')
    Returns: header (row 1) and every value in that column."""
    excel_data = _state["excel_data"]
    if not excel_data:
        return "ERROR: No spreadsheet loaded."

    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            col_cells = [c for c in sheet.cells if c.column == column]
            col_cells.sort(key=lambda c: int(c.row))

            if not col_cells:
                return f"Column {column} is empty in {sheet_name}."

            lines = [f"Column {column} in '{sheet_name}':"]
            for cell in col_cells:
                label = "HEADER" if cell.row == "1" else f"  Row {cell.row}"
                lines.append(f"  {label}: {cell.data_excel}")

            first_data = col_cells[1].row if len(col_cells) > 1 else col_cells[0].row
            last_data = col_cells[-1].row
            lines.append(f"\n  Data range: {column}{first_data}:{column}{last_data}")
            lines.append(f"  Total values: {len(col_cells)}")
            return "\n".join(lines)

    return f"Sheet not found: {sheet_name}"


@tool
def get_row_data(sheet_name: str, row: str) -> str:
    """Get all values in a specific row with column letters.
    Use row='1' to see headers (column names).
    Args:
        sheet_name: name of the sheet
        row: row number as string (e.g., '1' for headers)
    Returns: every cell value in that row."""
    excel_data = _state["excel_data"]
    if not excel_data:
        return "ERROR: No spreadsheet loaded."

    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            row_cells = [c for c in sheet.cells if c.row == row]
            row_cells.sort(key=lambda c: c.column)

            if not row_cells:
                return f"Row {row} is empty in {sheet_name}."

            label = "Headers" if row == "1" else f"Row {row}"
            lines = [f"{label} in '{sheet_name}':"]
            for cell in row_cells:
                lines.append(f"  [{cell.column}] = {cell.data_excel}")
            return "\n".join(lines)

    return f"Sheet not found: {sheet_name}"


@tool
def get_data_bounds(sheet_name: str) -> str:
    """Get the exact boundaries of data in a sheet.
    Returns: first row, last row, first column, last column, and headers.
    CRITICAL for writing correct formula ranges like B2:B50."""
    excel_data = _state["excel_data"]
    if not excel_data:
        return "ERROR: No spreadsheet loaded."

    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            if not sheet.cells:
                return f"Sheet '{sheet_name}' is empty."

            rows = sorted(set(int(c.row) for c in sheet.cells))
            cols = sorted(set(c.column for c in sheet.cells))

            # Get headers (row 1)
            headers = {c.column: c.data_excel for c in sheet.cells if c.row == "1"}

            lines = [
                f"Data bounds for '{sheet_name}':",
                f"  Rows: {rows[0]} to {rows[-1]} ({len(rows)} rows)",
                f"  Columns: {cols[0]} to {cols[-1]} ({len(cols)} columns)",
                f"  Total cells: {len(sheet.cells)}",
                f"",
                f"  Headers (row 1):",
            ]
            for col in cols:
                h = headers.get(col, "(empty)")
                lines.append(f"    [{col}] = {h}")

            data_start = rows[1] if len(rows) > 1 else rows[0]
            lines.append(f"")
            lines.append(f"  Data rows: {data_start} to {rows[-1]}")
            lines.append(f"  Example range for column B: B{data_start}:B{rows[-1]}")

            return "\n".join(lines)

    return f"Sheet not found: {sheet_name}"


# ================================================================
# SECTION 7: TOOLS — SHARED (all specialists can use)
# ================================================================

@tool
def show_excel_history(check: str = "all") -> str:
    """Show all changes made to the spreadsheet so far.
    Displays: what was changed, old value, new value, and when."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    return mgr.history()


@tool
def save_excel_copy(output_path: str) -> str:
    """Save the spreadsheet to a NEW file path (keeps original unchanged).
    Args:
        output_path: full path for the new file (e.g., 'sales_v2.xlsx')"""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    return mgr.save_as(output_path)


@tool
def reload_excel(confirm: str = "yes") -> str:
    """Reload the spreadsheet from the original file on disk.
    This DISCARDS all in-memory changes that haven't been auto-saved.
    Pass confirm='yes' to proceed."""
    if confirm.lower() != "yes":
        return "Reload cancelled. Pass confirm='yes' to proceed."
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    return mgr.reload()


# ================================================================
# SECTION 8: SPECIALIST PROMPTS + TOOLSETS
# ================================================================
# Each specialist gets ONLY the tools it needs. This prevents a
# specialist from doing things outside its role. role/goal/backstory
# are folded into one system prompt per specialist (see SECTION 8B).
# Excel context (sheet/cell addressing conventions) is appended to
# every specialist's prompt so it doesn't need a separate per-request
# Task description the way the old CrewAI version did.

_EXCEL_CONTEXT = (
    "\n\nEXCEL CONTEXT:\n"
    "- Data is in sheets. Use list_sheets to see them.\n"
    "- Cells are addressed as SheetName[ColumnRow], e.g., Sheet1[B3]\n"
    "- Row 1 usually has headers. Data starts from row 2.\n"
    "- Columns use letters (A, B, C...), rows use numbers (1, 2, 3...).\n"
    "- add_cell and batch_edit_excel's 'add' action are safe to call even "
    "when you're not certain a cell is empty: if the target position "
    "already holds a value, it gets UPDATED in place rather than "
    "creating a duplicate entry. You don't need to pre-check purely to "
    "avoid a collision — but searching first is still the right move to "
    "confirm you're targeting the correct cell in the first place.\n"
    "- Row and column values are plain strings ('5', 'B'), not numbers — "
    "always reuse the exact values returned by list_sheets/"
    "get_data_bounds/search_spreadsheet rather than reformatting or "
    "guessing them."
)

_DATA_SUFFICIENCY_POLICY = (
    "\n\nDATA SUFFICIENCY POLICY — read this before acting:\n"
    "Missing, empty, or limited data is NEVER permission to invent, add, "
    "or fabricate information on your own initiative. Only call add_cell, "
    "add_row, or batch_edit_excel's 'add' action when the user has "
    "explicitly asked for an insertion, creation, or population operation "
    "in THIS request.\n"
    "- If the sheet is completely empty (no rows, or only headers with no "
    "data): do not create rows to fill it. Tell the user the spreadsheet "
    "has no usable data, explain that the requested operation can't be "
    "performed without a source, and ask what they want to do next — add "
    "new information, provide another file, or something else.\n"
    "- If the sheet has some data but it's too limited, incomplete, or not "
    "relevant to the request: do not fill the gap with invented or "
    "assumed values. State exactly what you found and why it's not "
    "enough, then ask whether to proceed with the limited data, provide "
    "more information, or use another file.\n"
    "- If what you found doesn't meaningfully relate to the request: say "
    "so explicitly instead of forcing unrelated data into the answer, "
    "state what information you'd need, and ask for it.\n"
    "- Never overwrite, delete, or replace existing cell values, and "
    "never invent values for missing fields, unless the user explicitly "
    "asked for that in this request.\n"
    "- When you can't proceed, your final answer must cover: (1) what "
    "data you found, (2) what's missing/insufficient/irrelevant, (3) why "
    "that blocks the request, (4) what you need to proceed, (5) the "
    "choices the user can make. Do not silently call a tool with assumed "
    "or invented values instead of asking."
)

_READER_PROMPT = (
    "ROLE: Spreadsheet Reader\n"
    "GOAL: Read and search spreadsheet content accurately. Use keyword "
    "search for exact cell matches, vector search for meaning-based "
    "queries across rows and columns.\n\n"
    "You are the data analyst. You know every sheet, row, and column. "
    "Your workflow: first list_sheets to see what exists, then "
    "search_spreadsheet for exact matches or vector_search_excel "
    "for conceptual/meaning-based queries — there is no 'view "
    "everything' tool, since a large sheet can have lakhs of rows "
    "and would never fit in context. Always search instead of "
    "trying to see the whole sheet at once. "
    "Excel data is organized as Sheet -> Row -> Column -> Cell."
    + _EXCEL_CONTEXT
    + _DATA_SUFFICIENCY_POLICY
)
_READER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets]

_WRITER_PROMPT = (
    "ROLE: Spreadsheet Writer\n"
    "GOAL: Add new data to the spreadsheet at the correct position. "
    "Always check the sheet structure first before adding.\n\n"
    "You are the data entry specialist. Before adding anything, you "
    "ALWAYS use list_sheets to see sheet names, then search_spreadsheet "
    "or vector_search_excel to understand the structure (headers in "
    "row 1, data below) — you never try to view the whole sheet, "
    "since it may hold lakhs of rows. "
    "You can add individual cells or entire rows. "
    "Columns use letters (A, B, C...), rows use numbers (1, 2, 3...). "
    "For adding MORE THAN 2-3 cells/rows at once (e.g. '20 dummy "
    "rows'), use batch_edit_excel with all the edits in one call "
    "instead of calling add_cell/add_row repeatedly — that one call "
    "actually performs and saves every edit, so it can't end up "
    "half-described and half-done. "
    "IMPORTANT: describing the new data in your final answer does NOT "
    "save it. The file only changes when you actually call add_cell, "
    "add_row, or batch_edit_excel. You are not done until that tool "
    "call has happened."
    + _EXCEL_CONTEXT
    + _DATA_SUFFICIENCY_POLICY
)
_WRITER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets,
                  add_cell, add_row, batch_edit_excel, show_excel_history]

_UPDATER_PROMPT = (
    "ROLE: Spreadsheet Updater\n"
    "GOAL: Modify existing cell values accurately. Find the exact cell "
    "first, then update it.\n\n"
    "You are the data editor. You never update blindly. Your workflow: "
    "1) Search for the cell to change, 2) Note the exact sheet name, "
    "row number, and column letter, 3) Update with the new value, "
    "4) Verify the change. For find/replace across many cells, use "
    "replace_all_excel. For any other multi-cell change (e.g. "
    "'update rows 6-15 in column B'), use batch_edit_excel with all "
    "the edits in one call — do not try to call update_cell many "
    "times in a row, since that has been observed to end with the "
    "model just describing the changes instead of making them. "
    "Cell positions look like: SheetName[ColumnRow] e.g., Sheet1[B3]. "
    "IMPORTANT: describing the new value in your final answer does NOT "
    "save it — the file only changes when you actually call "
    "update_cell, replace_all_excel, or batch_edit_excel. You are not "
    "done until every changed cell has a matching tool call."
    + _EXCEL_CONTEXT
    + _DATA_SUFFICIENCY_POLICY
)
_UPDATER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets,
                   update_cell, replace_all_excel, batch_edit_excel, show_excel_history]

_DELETER_PROMPT = (
    "ROLE: Spreadsheet Deleter\n"
    "GOAL: Remove specific cells or rows from the spreadsheet safely. "
    "Always confirm the exact position before deleting.\n\n"
    "You are the data cleanup specialist. Deletion is permanent, "
    "so you are extra careful. Your workflow: 1) Search for the "
    "content to delete with search_spreadsheet or vector_search_excel "
    "to confirm it's the right cell/row (never try to view the "
    "whole sheet — it may hold lakhs of rows), 2) Delete using the "
    "exact sheet, row, column. "
    "You can delete a single cell or an entire row. For deleting "
    "MORE THAN 2-3 cells/rows at once, use batch_edit_excel with all "
    "the deletions in one call instead of calling delete_cell/"
    "delete_row repeatedly. "
    "IMPORTANT: you are not done until delete_cell, delete_row, or "
    "batch_edit_excel has actually been called — saying a cell/row "
    "was removed in your final answer does not remove it."
    + _EXCEL_CONTEXT
)
_DELETER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets,
                   delete_cell, delete_row, batch_edit_excel, show_excel_history]

_FORMULA_PROMPT = (
    "ROLE: Formula Generator\n"
    "GOAL: Generate and insert correct Excel formulas into the "
    "spreadsheet. Understand the data layout first, then craft the "
    "precise formula. Support all Excel functions: SUM, AVERAGE, "
    "COUNT, COUNTIF, SUMIF, IF, VLOOKUP, INDEX, MATCH, MIN, MAX, "
    "CONCATENATE, and more.\n\n"
    "You are an Excel formula expert. You NEVER guess cell ranges. "
    "Your strict workflow:\n"
    "1) Use get_data_bounds to see the sheet's exact row/column boundaries\n"
    "2) Use get_row_data with row='1' to read the column headers\n"
    "3) Use get_column_data to inspect the actual values in target columns\n"
    "4) Craft the formula using the EXACT ranges from what you found\n"
    "5) Use insert_formula to place it in the right cell\n\n"
    "Example: User says 'sum of sales column'.\n"
    "  → get_data_bounds('Sheet1') → data rows 2 to 50, Sales is column D\n"
    "  → insert_formula('Sheet1', '51', 'D', '=SUM(D2:D50)')\n\n"
    "You know all Excel functions and can build complex nested formulas. "
    "For VLOOKUP, always check both the lookup sheet and data sheet first. "
    "IMPORTANT: writing the formula in your final answer does NOT "
    "insert it — the file only changes when you actually call "
    "insert_formula. You are not done until that tool call has happened."
    + _EXCEL_CONTEXT
)
_FORMULA_TOOLS = [list_sheets, search_spreadsheet, get_data_bounds,
                   get_column_data, get_row_data, insert_formula, show_excel_history]

_SPECIALIST_DEFS = {
    "reader": (_READER_PROMPT, _READER_TOOLS),
    "writer": (_WRITER_PROMPT, _WRITER_TOOLS),
    "updater": (_UPDATER_PROMPT, _UPDATER_TOOLS),
    "deleter": (_DELETER_PROMPT, _DELETER_TOOLS),
    "formula_gen": (_FORMULA_PROMPT, _FORMULA_TOOLS),
}


# ================================================================
# SECTION 8B: SPECIALIST SUBGRAPHS (ReAct agent ↔ tools loop)
# ================================================================

class SpecialistState(TypedDict):
    """Local state for one specialist's tool-calling subgraph.
    Only `messages` — the field every specialist subgraph shares
    with the parent crew graph (see CrewState in SECTION 9)."""
    messages: Annotated[list, add_messages]


def _build_specialist_subgraph(node_name: str, system_prompt: str, tools: list, llm):
    """
    Build one specialist's ReAct loop (agent ↔ tools) as its OWN
    compiled StateGraph — a genuine LangGraph subgraph, not an inline
    function. It gets added directly as a node into the parent crew
    graph in build_crew_graph() below; because both graphs use the
    same `messages: Annotated[list, add_messages]` field, LangGraph
    invokes the subgraph with the parent's message history and merges
    its output straight back in — no glue code required.
    """
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: SpecialistState) -> SpecialistState:
        # The system prompt is injected fresh on every call rather
        # than stored permanently in `messages` — several specialists
        # can run back-to-back on the SAME shared message history
        # (see route_agents), and each one needs to reason under ITS
        # OWN role prompt without an earlier specialist's system
        # message still sitting at the front of the conversation.
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
    """
    Compile the 5 specialist subgraphs.

    No manager agent — routing between them is done in plain Python
    by route_agents() (see SECTION 9), not by an LLM.

        ├── Reader        (view, search, vector search, list sheets)
        ├── Writer        (add cells, add rows)
        ├── Updater       (update cells, replace all)
        ├── Deleter       (delete cells, delete rows)
        └── Formula Gen   (inspect data, generate & insert formulas)
    """
    return {
        name: _build_specialist_subgraph(name, prompt, tools, llm)
        for name, (prompt, tools) in _SPECIALIST_DEFS.items()
    }


# ================================================================
# SECTION 9: CREW GRAPH BUILDER
# ================================================================

# Checked in this fixed priority order so a compound request like
# "delete the outdated total and add a SUM formula instead" still
# hits every specialist it needs, instead of only the first verb an
# LLM manager's own reasoning happened to notice. Delete is checked
# first, formulas next (formula phrasing often also contains "add"/
# "insert", which should NOT route to the plain Writer), then update,
# then plain add/insert.
_INTENT_ROUTES = [
    ("deleter", ("delete", "remove")),
    ("formula_gen", ("formula", "sum", "average", "count", "calculate",
                      "total", "vlookup", "sumif", "countif", "index",
                      "match")),
    ("updater", ("update", "edit", "modify", "change", "replace",
                 "fix", "correct", "rename", "space", "spacing",
                 "blank", "gap", "format", "formatting", "indent",
                 "align")),
    ("writer", ("add", "insert", "append", "fill", "set ")),
]


class CrewState(TypedDict):
    """Parent graph state — shared with every specialist subgraph via
    `messages`, plus a `pending` queue of specialist names still to run."""
    messages: Annotated[list, add_messages]
    pending: List[str]


def route_agents(user_request: str) -> List[str]:
    """
    Fast, code-based router — replaces the old CrewAI manager Agent.

    HIERARCHICAL made a manager reason (via its own LLM call) about
    who to delegate to, then review the specialist's output afterward
    — two extra LLM round trips on top of the actual work, on every
    single request. This does the same routing with a keyword check,
    which costs ~0ms instead of an LLM call, and returns an ORDERED
    list of specialist names so multi-verb requests still get every
    specialist they need (see _INTENT_ROUTES above).

    Every specialist already carries search_spreadsheet/vector_search_excel/
    list_sheets and is instructed (see its system prompt) to look up
    the exact sheet/cell itself before acting — so a dedicated Reader
    hop is only used as the fallback below, when nothing else
    matched, not as a mandatory first step for the write specialists.
    """
    text = user_request.lower()
    matched = [name for name, keywords in _INTENT_ROUTES
               if any(kw in text for kw in keywords)]
    if matched:
        return matched

    # _INTENT_ROUTES can never cover every real phrasing for a write
    # request. _WRITE_INTENT_KEYWORDS is kept deliberately broader for
    # exactly this reason — if IT thinks this looks like a write request
    # even though _INTENT_ROUTES didn't recognize it, send it to the
    # Updater (which carries search tools AND update_cell/
    # replace_all_excel/batch_edit_excel) rather than falling back to
    # the read-only Reader, which has no way to act on it at all.
    if _looks_like_write_request(user_request):
        return ["updater"]

    return ["reader"]


def _route_node(state: CrewState) -> CrewState:
    """Reads the latest human message and decides the specialist queue."""
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
    )
    text = last_human.content if last_human else ""
    if not isinstance(text, str):
        text = str(text)
    return {"pending": route_agents(text)}


def _advance_node(state: CrewState) -> CrewState:
    """Pops the specialist that just ran off the pending queue."""
    return {"pending": state["pending"][1:]}


def _dispatch(state: CrewState) -> str:
    """Conditional-edge function shared by 'route' and 'advance': go to
    the next pending specialist by name, or END when the queue is empty."""
    pending = state.get("pending") or []
    return pending[0] if pending else END


def build_crew_graph(specialists: Dict):
    """
    Build the parent LangGraph that replaces CrewAI's Crew.

    - route() (plain Python, no LLM call) decides which specialist(s)
      handle a request — see route_agents()'s docstring for why this
      replaced an LLM manager.
    - Each specialist in `specialists` (from build_specialists) is
      added as a SUBGRAPH node. When more than one is matched (e.g.
      "find all sales below 100 and delete them"), they run one after
      another on the SAME shared `messages` list, so the second
      specialist sees the first one's tool calls/results as context
      automatically — the LangGraph equivalent of CrewAI's
      sequential-process context passing, with no manager LLM call
      anywhere in the loop.
    - No separate "memory" layer and no verbose reasoning traces are
      added here on purpose — same reasoning as under CrewAI: the
      project already has its own persistent, local, offline RAG
      layer (Chroma) for file content, so nothing else is needed, and
      printing every reasoning step to the console cost real
      wall-clock time on a long multi-agent run.
    - Compiled ONCE per ExcelAgentSystem instance (not rebuilt per
      request) — unlike a CrewAI Crew, a LangGraph graph's structure
      doesn't depend on the user's request text, only route()'s
      runtime decision does.
    """
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


def _extract_text(content) -> str:
    """Extract plain text from an LLM message's content.
    Handles both plain strings and structured content parts (list of
    dicts with 'type'/'text' keys) returned by newer
    langchain-google-genai versions."""
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
    """Find the last AI message that actually has visible text (i.e.
    the specialist's final answer, not an intermediate tool-call-only
    message with empty content), across however many specialists ran."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            text = _extract_text(m.content).strip()
            if text:
                return text
    return "(no response)"


# ================================================================
# SECTION 10: MAIN SYSTEM (ties everything together)
# ================================================================
def setup_llm():
    """
    Get a working LLM.
    - If credentials are saved on disk → load and use them.
    - If not → ask once, validate, save for next time.
    """
    # ---- Saved credentials? Use them ----
    if has_credentials():
        creds = load_credentials()
        info(f"Using saved credentials (model: {creds['model']})")
        return get_agentic_llm(creds["api_key"], creds["model"])

    # ---- First run: ask the user ----
    header("First-time setup")

    while True:
        api_key = prompt_input("Gemini API key: ")
        if not api_key:
            warn("API key cannot be empty.\n")
            continue

        info("Available models:")
        for i, m in enumerate(SUPPORTED_MODELS, 1):
            print(f"    {i}. {m}")

        choice = prompt_input(f"\nPick a model (1-{len(SUPPORTED_MODELS)}): ")
        try:
            model = SUPPORTED_MODELS[int(choice) - 1]
        except (ValueError, IndexError):
            warn("Invalid choice.\n")
            continue

        # ---- Test before saving ----
        spinner = Spinner(f"Testing connection to {model}...", done_label="Connected — credentials saved").start()
        result = validate_api_key(api_key, model)
        spinner.stop(ok=result["success"])

        if result["success"]:
            save_credentials(api_key, model)
            return get_agentic_llm(api_key, model)

        error(f"{result['error']}\n")
class ExcelAgentSystem:
    """
    The main orchestrator for Excel files.

    Takes a filepath, processes it, and runs a chatbot loop
    where a LangGraph crew (router + specialist subgraphs) handles
    spreadsheet operations.

    Usage:
        system = ExcelAgentSystem("sales.xlsx")
        system.chat()   # interactive loop
        # OR
        result = system.process_request("find all rows with revenue > 1000")
    """

    def __init__(self, filepath: str, llm=None):
        """
        Initialize the system.

        Args:
            filepath: path to the .xlsx file
            llm: (optional) LangChain chat model instance for the
                 specialists (must support .bind_tools(), e.g.
                 ChatGoogleGenerativeAI). Defaults to Gemini via the
                 project's saved credentials.
        """
        self.filepath = filepath

        # ---- Set up the LLM ----
        if llm:
            self.llm = llm
        else:
            # Default: load saved Gemini credentials and build a
            # LangChain ChatGoogleGenerativeAI tuned for agentic tool
            # calling (see get_agentic_llm's docstring for why plain
            # get_llm() defaults aren't safe for a multi-tool loop).
            cred = load_credentials()
            if not cred:
                raise RuntimeError(
                    "No saved Gemini credentials found. Run onboarding first "
                    "(or pass an explicit llm= to ExcelAgentSystem)."
                )
            info(f"Using saved credentials (model: {cred['model']})")
            self.llm = get_agentic_llm(cred["api_key"], cred["model"])

        # ---- Step 1: Process the file ----
        header("DOXL AI — Excel Agent")
        process_file(filepath)

        # ---- Step 2: Build specialist subgraphs + parent crew graph ----
        self.specialists = build_specialists(self.llm)
        self.graph = build_crew_graph(self.specialists)

        header(
            "Ready",
            "Reader · Writer · Updater · Deleter · Formula Gen",
        )
        print(
            "  Ask in plain English, e.g. \"find rows where revenue > 1000\".\n"
            "  'sheets' lists sheet names, 'history' shows changes, 'quit' exits.\n"
            "  Full-sheet 'view' is disabled — sheets can hold lakhs of rows."
        )

    def process_request(self, user_input: str) -> str:
        """
        Send one request through the LangGraph crew and get the result.

        route() picks the right specialist(s), each runs its own
        agent ↔ tools loop, and the result is returned. If anything in
        this request wrote to the sheet, the (expensive) vector index
        is rebuilt exactly once here — after the whole request
        finishes — rather than after every individual tool call.

        A ground-truth check is appended after the graph finishes. The
        graph's text answer is the LLM's own claim about what it did —
        and that claim has been observed to be wrong (a final answer
        listing 20 "added" rows with zero actual add_cell/add_row/
        batch_edit_excel calls behind it, because the model generated
        a description instead of using its tools). mgr.changelog only
        grows when a write tool is actually invoked, so comparing its
        length before/after this request gives an honest, code-verified
        answer regardless of what the graph's text says.
        """
        mgr = _state.get("excel_manager")
        changes_before = len(mgr.changelog.changes) if mgr else 0

        result_state = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "pending": []},
            config={"recursion_limit": 50},
        )
        rebuild_excel_vector_index_if_dirty()

        mgr = _state.get("excel_manager")
        changes_after = len(mgr.changelog.changes) if mgr else 0
        made = changes_after - changes_before

        result_str = _last_answer_text(result_state["messages"])
        if made > 0:
            result_str += f"\n\n[VERIFIED] {made} change(s) were actually written to the file."
        elif _looks_like_write_request(user_input):
            result_str += (
                "\n\n[WARNING] This looks like an add/update/delete/formula "
                "request, but no write tool was actually called — the file "
                "was NOT modified, regardless of what the summary above "
                "says. Try rephrasing more explicitly, or ask again."
            )
        return result_str

    def chat(self):
        """
        Interactive chatbot loop.

        Keeps running until the user types 'quit'.
        Each request goes through the LangGraph crew.
        """
        while True:
            # ---- Get user input ----
            try:
                user_input = prompt_input("\nYou: ")
            except (KeyboardInterrupt, EOFError):
                print()
                success("Goodbye!")
                break

            if not user_input:
                continue

            # ---- Check for exit ----
            if user_input.lower() in ("quit", "exit", "q", "bye"):
                success("Goodbye!")
                break

            # ---- Quick shortcuts (skip the graph for speed) ----
            if user_input.lower() == "view":
                warn(
                    "'view' is disabled — a sheet can have lakhs of rows, "
                    "which won't fit in context. Try 'sheets' to see what's "
                    "there, or search naturally, e.g. 'find rows with status "
                    "= pending'."
                )
                continue

            if user_input.lower() == "sheets":
                print(list_sheets.invoke("all"))
                continue

            if user_input.lower() == "history":
                print(show_excel_history.invoke("all"))
                continue

            # ---- Process through the LangGraph crew ----
            spinner = Spinner("Thinking...").start()
            try:
                result = self.process_request(user_input)
                spinner.stop(ok=False)  # request done — result is shown below, no extra success line
                agent_result(result)

            except Exception as e:
                spinner.stop(ok=False)
                error(str(e))
                info("Try rephrasing your request.")


# ================================================================
# SECTION 11: ENTRY POINT
# ================================================================

def main():
    """CLI entry point: python excel_agent.py <filepath.xlsx>"""

    if len(sys.argv) < 2:
        print("Usage: python excel_agent.py <filepath.xlsx>")
        print("Example: python excel_agent.py sales_data.xlsx")
        sys.exit(1)

    filepath = sys.argv[1]

    # ---- Configure LLM (change this to use a different model) ----
    #
    # Default: None → loads saved Gemini credentials via
    # pipeline.config.get_agentic_llm (see ExcelAgentSystem.__init__).
    #
    # Any other LangChain chat model that supports .bind_tools() also
    # works, e.g.:
    #   from langchain_openai import ChatOpenAI
    #   llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    #
    llm = None  # None = uses default (saved Gemini credentials)

    system = ExcelAgentSystem(filepath, llm=llm)
    system.chat()


if __name__ == "__main__":
    main()
