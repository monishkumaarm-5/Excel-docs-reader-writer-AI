# excel_agent.py
"""
LangGraph Multi-Agent Excel Handler
====================================

Architecture:
    Code Router (route_agents — keyword-based, no LLM call)
        ├── Reader Subgraph        → list sheets, exact search, RAG semantic search
        ├── Writer Subgraph        → add cells, add rows
        ├── Updater Subgraph       → update cells, find-replace
        ├── Deleter Subgraph       → delete cells, delete rows
        └── Formula Gen Subgraph   → inspect data, craft & insert Excel formulas

Flow:
    1. User gives filepath → file is read, chunked, vectorized
    2. Chatbot loop starts
    3. User types a request
    4. route() node picks the specialist(s) by keyword — no LLM call
    5. Each specialist subgraph runs in turn (agent ↔ tools loop),
       sharing one growing `messages` list → result returned
    6. Loop continues until user quits
"""

import os
import sys
import json
from collections import deque
from typing import Optional, List, Dict, TypedDict, Annotated, Union

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.tools import tool
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

from doxl_ai_terminal.Frontier.changelogmanager_excel import LiveExcelManager
from doxl_ai_terminal.Frontier.fileReader import read_excel
from doxl_ai_terminal.Chunker.chunking import chunk_excel_row_wise, chunk_excel_column_wise
from doxl_ai_terminal.data_handler.vector_config import store_in_chroma, make_collection_name
from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager
from doxl_ai_terminal.data_structure.excel import parse_row
from doxl_ai_terminal.pipeline.config import SUPPORTED_MODELS, validate_api_key, get_agentic_llm
from doxl_ai_terminal.pipeline.credential_store import (
    has_credentials, load_credentials, save_credentials,
)
from doxl_ai_terminal.pipeline.terminal_ui import (
    Spinner, TaskProgress, info, success, warn, error, header, agent_result,
    prompt_input, choice_prompt, console,
)


# ================================================================
# SECTION 1: SHARED STATE
# ================================================================

_state: Dict = {
    "excel_manager": None,
    "excel_data": None,
    "vector_managers": {},
    "collections": [],
    "filepath": None,
    "_dirty": False,
    "task_queue": deque(),
}


# ================================================================
# SECTION 2: FILE PROCESSOR
# ================================================================

def process_file(filepath: str):
    """Full pipeline: Read → Chunk → Vector store → Load editor."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    ext = os.path.splitext(filepath)[1].lower()
    if ext != ".xlsx":
        raise ValueError(f"Only .xlsx files supported. Got: {ext}")

    _state["filepath"] = filepath
    _state["task_queue"] = deque()

    with Spinner("Loading the file..."):
        excel_data = read_excel(filepath)
        _state["excel_data"] = excel_data

        sheet_names = [s.sheet_name for s in excel_data.sheets]
        total_cells = sum(len(s.cells) for s in excel_data.sheets)

        _state["excel_manager"] = LiveExcelManager(filepath)
        _rebuild_excel_vector_index(filepath, excel_data)

    info(f"{len(sheet_names)} sheet(s): {', '.join(sheet_names)}")
    success(f"Ready! {total_cells} cells across {len(sheet_names)} sheet(s).\n")


# ================================================================
# SECTION 2B: RE-INDEX
# ================================================================

CHUNK_STRATEGIES = [
    ("row_wise", chunk_excel_row_wise),
    ("column_wise", chunk_excel_column_wise),
]


def _rebuild_excel_vector_index(filepath: str, excel_data) -> None:
    """(Re)build every chunking strategy's Chroma collection."""
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
            pass

        store_in_chroma(chunks, collection_name)
        collections.append(collection_name)
        vector_managers[format_name] = VectorDBManager().load_collection(collection_name)

    _state["collections"] = collections
    _state["vector_managers"] = vector_managers


def _refresh_excel_data() -> None:
    """Sync _state["excel_data"] with the manager's live in-memory data."""
    mgr = _state.get("excel_manager")
    if mgr is None:
        return
    _state["excel_data"] = mgr.excel_data
    _state["_dirty"] = True


def mark_excel_dirty_and_refresh() -> None:
    """Called by every mutating tool right after it changes the sheet."""
    try:
        _refresh_excel_data()
    except Exception as e:
        warn(f"Refreshing sheet data failed: {e}")


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

    mgr = _state.get("excel_manager")
    if mgr is not None:
        mgr._sync_to_file()

    if _task_progress:
        _task_progress.finish(saved=True)
        _task_progress = None


def rebuild_excel_vector_index_if_dirty() -> None:
    """Rebuild Chroma vector index if something changed."""
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


MAX_RELEVANT_DISTANCE = 1.3


@tool
def vector_search_excel(query: str) -> str:
    """Semantic search — find content by MEANING, not just exact words.
    Searches across every chunking strategy (row-wise AND column-wise).
    Example: searching 'profit' also finds 'revenue', 'margin', 'earnings'.
    Returns the top matches or says if nothing relevant was found."""
    managers = _state.get("vector_managers") or {}
    if not managers:
        return "ERROR: Vector store not loaded."

    candidates = []
    for format_name, vm in managers.items():
        try:
            candidates.extend(vm.search_with_scores(query, k=4))
        except Exception as e:
            return f"Search error ({format_name}): {e}"

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
def add_cell(sheet_name: str, row: int, column: str, value: Union[str, int, float]) -> str:
    """Add a new cell to the spreadsheet. If the target position already
    holds a value, it is UPDATED in place instead of creating a duplicate.
    Args:
        sheet_name: name of the sheet (use list_sheets to see names)
        row: row number as integer (e.g., 5)
        column: column letter (e.g., 'A', 'B', 'C')
        value: the text/number to put in the cell
    Queued and saved once the whole request finishes."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    row = parse_row(row)
    result = mgr.add_cell(sheet_name, row, column, value, auto_save=False)
    _queue_task(f"add_cell {sheet_name}[{column}{row}]")
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
    Queued and saved once the whole request finishes."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."

    row_data = {}
    try:
        for pair in row_data_str.split(","):
            col, val = pair.strip().split("=", 1)
            row_data[col.strip()] = val.strip()
    except ValueError:
        return "ERROR: Format must be 'A=value1,B=value2,C=value3'"

    result = mgr.add_row(sheet_name, row_data, auto_save=False)
    _queue_task(f"add_row {sheet_name}")
    mark_excel_dirty_and_refresh()
    return result


# ================================================================
# SECTION 5: TOOLS — UPDATER
# ================================================================

@tool
def update_cell(sheet_name: str, row: int, column: str, new_value: Union[str, int, float]) -> str:
    """Update an existing cell in the spreadsheet.
    Args:
        sheet_name: name of the sheet
        row: row number as integer (e.g., 3)
        column: column letter (e.g., 'B')
        new_value: the new value to put in the cell
    IMPORTANT: Use search_spreadsheet first to find the exact position.
    Queued and saved once the whole request finishes."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    row = parse_row(row)
    result = mgr.update_cell(sheet_name, row, column, new_value, auto_save=False)
    _queue_task(f"update_cell {sheet_name}[{column}{row}]")
    mark_excel_dirty_and_refresh()
    return result


@tool
def replace_all_excel(old_text: str, new_text: str) -> str:
    """Find and replace text across ALL cells in ALL sheets.
    Args:
        old_text: the text to find
        new_text: the text to replace it with
    Queued and saved once the whole request finishes."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    result = mgr.replace_all(old_text, new_text, auto_save=False)
    _queue_task("replace_all_excel")
    mark_excel_dirty_and_refresh()
    return result


@tool
def batch_edit_excel(edits_json: str) -> str:
    """Apply MULTIPLE add/update/delete edits in ONE call, with a single
    file save at the end.

    Args:
        edits_json: a JSON array of edit objects, e.g.:
          [
            {"action": "add", "sheet": "Sheet1", "row": 6, "column": "A", "value": "Dummy 6"},
            {"action": "update", "sheet": "Sheet1", "row": 3, "column": "B", "value": "New"},
            {"action": "delete", "sheet": "Sheet1", "row": 9, "column": "C"}
          ]
        "value" is required for add/update and ignored for delete.
        Row numbers should be integers.
    Saved once for the whole request."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."

    try:
        edits = json.loads(edits_json)
    except json.JSONDecodeError as e:
        return f"ERROR: edits_json must be a valid JSON array. {e}"

    if not isinstance(edits, list) or not edits:
        return "ERROR: edits_json must be a non-empty JSON array."

    normalized = []
    for e in edits:
        if not isinstance(e, dict) or "action" not in e:
            return f"ERROR: every edit needs an 'action' key. Bad edit: {e}"
        normalized.append({
            "action": e.get("action"),
            "sheet": e.get("sheet"),
            "row": parse_row(e.get("row", 0)),
            "column": e.get("column"),
            "value": e.get("value", ""),
        })

    results = mgr.batch_edit(normalized, auto_save=False)
    _queue_task(f"batch_edit_excel ({len(normalized)} edits)")
    mark_excel_dirty_and_refresh()
    return f"Applied {len(results)} edit(s):\n" + "\n".join(results)


# ================================================================
# SECTION 6: TOOLS — DELETER
# ================================================================

@tool
def delete_cell(sheet_name: str, row: int, column: str) -> str:
    """Delete a specific cell from the spreadsheet.
    Args:
        sheet_name: name of the sheet
        row: row number as integer (e.g., 3)
        column: column letter (e.g., 'B')
    IMPORTANT: Use search_spreadsheet first to confirm the exact cell.
    Queued and saved once the whole request finishes."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    row = parse_row(row)
    result = mgr.delete_cell(sheet_name, row, column, auto_save=False)
    _queue_task(f"delete_cell {sheet_name}[{column}{row}]")
    mark_excel_dirty_and_refresh()
    return result


@tool
def delete_row(sheet_name: str, row: int) -> str:
    """Delete an entire row (all cells in that row) from a sheet.
    Args:
        sheet_name: name of the sheet
        row: row number as integer (e.g., 5)
    WARNING: This removes ALL cells in the row.
    Queued and saved once the whole request finishes."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    row = parse_row(row)
    result = mgr.delete_row(sheet_name, row, auto_save=False)
    _queue_task(f"delete_row {sheet_name}[{row}]")
    mark_excel_dirty_and_refresh()
    return result


# ================================================================
# SECTION 6B: TOOLS — FORMULA GENERATOR
# ================================================================

@tool
def insert_formula(sheet_name: str, row: int, column: str, formula: str) -> str:
    """Insert an Excel formula into a specific cell.
    Args:
        sheet_name: name of the sheet
        row: row number as integer (e.g., 10)
        column: column letter (e.g., 'C')
        formula: the Excel formula starting with '=' (e.g., '=SUM(B2:B9)')
    Queued and saved once the whole request finishes."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."

    row = parse_row(row)

    if not formula.startswith("="):
        formula = "=" + formula

    # Check if cell already exists → update it. Otherwise → add it.
    excel_data = _state["excel_data"]
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            for cell in sheet.cells:
                if cell.row == row and cell.column == column:
                    result = mgr.update_cell(sheet_name, row, column, formula, auto_save=False)
                    _queue_task(f"insert_formula {sheet_name}[{column}{row}]")
                    mark_excel_dirty_and_refresh()
                    return f"Formula inserted (updated): {sheet_name}[{column}{row}] = {formula}\n{result}"

    result = mgr.add_cell(sheet_name, row, column, formula, auto_save=False)
    _queue_task(f"insert_formula {sheet_name}[{column}{row}]")
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
            col_cells.sort(key=lambda c: c.row)

            if not col_cells:
                return f"Column {column} is empty in {sheet_name}."

            lines = [f"Column {column} in '{sheet_name}':"]
            for cell in col_cells:
                label = "HEADER" if cell.row == 1 else f"  Row {cell.row}"
                lines.append(f"  {label}: {cell.data_excel}")

            first_data = col_cells[1].row if len(col_cells) > 1 else col_cells[0].row
            last_data = col_cells[-1].row
            lines.append(f"\n  Data range: {column}{first_data}:{column}{last_data}")
            lines.append(f"  Total values: {len(col_cells)}")
            return "\n".join(lines)

    return f"Sheet not found: {sheet_name}"


@tool
def get_row_data(sheet_name: str, row: int) -> str:
    """Get all values in a specific row with column letters.
    Use row=1 to see headers (column names).
    Args:
        sheet_name: name of the sheet
        row: row number as integer (e.g., 1 for headers)
    Returns: every cell value in that row."""
    excel_data = _state["excel_data"]
    if not excel_data:
        return "ERROR: No spreadsheet loaded."

    row = parse_row(row)

    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            row_cells = [c for c in sheet.cells if c.row == row]
            row_cells.sort(key=lambda c: c.column)

            if not row_cells:
                return f"Row {row} is empty in {sheet_name}."

            label = "Headers" if row == 1 else f"Row {row}"
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

            rows = sorted({c.row for c in sheet.cells})
            cols = sorted({c.column for c in sheet.cells})

            headers = {c.column: c.data_excel for c in sheet.cells if c.row == 1}

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
# SECTION 7: TOOLS — SHARED
# ================================================================

@tool
def show_excel_history(check: str = "all") -> str:
    """Show all changes made to the spreadsheet so far."""
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
    This DISCARDS all in-memory changes. Pass confirm='yes' to proceed."""
    if confirm.lower() != "yes":
        return "Reload cancelled. Pass confirm='yes' to proceed."
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    _state.get("task_queue", deque()).clear()
    return mgr.reload()


# ================================================================
# SECTION 8: SPECIALIST PROMPTS + TOOLSETS
# ================================================================

_EXCEL_CONTEXT = (
    "\n\nEXCEL CONTEXT:\n"
    "- Data is in sheets. Use list_sheets to see them.\n"
    "- Cells are addressed as SheetName[ColumnRow], e.g., Sheet1[B3]\n"
    "- Row 1 usually has headers. Data starts from row 2.\n"
    "- Columns use letters (A, B, C...), rows use integers (1, 2, 3...).\n"
    "- Row numbers are integers, not strings — pass 5, not '5'.\n"
    "- Column letters are strings — pass 'B', not 2.\n"
    "- add_cell and batch_edit_excel's 'add' action are safe to call even "
    "when you're not certain a cell is empty: if the target position "
    "already holds a value, it gets UPDATED in place.\n"
)

_DATA_SUFFICIENCY_POLICY = (
    "\n\nDATA SUFFICIENCY POLICY — read this before acting:\n"
    "Missing, empty, or limited data is NEVER permission to invent, add, "
    "or fabricate information on your own initiative. Only call add_cell, "
    "add_row, or batch_edit_excel's 'add' action when the user has "
    "explicitly asked for an insertion, creation, or population operation "
    "in THIS request.\n"
    "- If the sheet is completely empty: tell the user and ask what to do.\n"
    "- If data is too limited or not relevant: state what you found and ask.\n"
    "- Never overwrite or invent values unless the user explicitly asked.\n"
    "- When you can't proceed, explain: (1) what data you found, (2) what's "
    "missing, (3) why that blocks the request, (4) what you need, (5) "
    "the choices the user can make."
)

_READER_PROMPT = (
    "ROLE: Spreadsheet Reader\n"
    "GOAL: Read and search spreadsheet content accurately. Use keyword "
    "search for exact cell matches, vector search for meaning-based queries.\n\n"
    "Your workflow: first list_sheets to see what exists, then "
    "search_spreadsheet for exact matches or vector_search_excel "
    "for conceptual queries. There is no 'view everything' tool — "
    "always search instead."
    + _EXCEL_CONTEXT
    + _DATA_SUFFICIENCY_POLICY
)
_READER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets]

_WRITER_PROMPT = (
    "ROLE: Spreadsheet Writer\n"
    "GOAL: Add new data to the spreadsheet at the correct position.\n\n"
    "Before adding, ALWAYS use list_sheets and search to understand the "
    "structure first. For MORE THAN 2-3 cells/rows, use batch_edit_excel "
    "instead of calling add_cell/add_row repeatedly. "
    "IMPORTANT: describing data in your final answer does NOT save it — "
    "the file only changes when you actually call add_cell, add_row, or "
    "batch_edit_excel."
    + _EXCEL_CONTEXT
    + _DATA_SUFFICIENCY_POLICY
)
_WRITER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets,
                  add_cell, add_row, batch_edit_excel, show_excel_history]

_UPDATER_PROMPT = (
    "ROLE: Spreadsheet Updater\n"
    "GOAL: Modify existing cell values accurately. Find the exact cell "
    "first, then update it.\n\n"
    "Your workflow: 1) Search for the cell, 2) Note the exact position, "
    "3) Update with new value. For find/replace, use replace_all_excel. "
    "For multi-cell changes, use batch_edit_excel in one call. "
    "IMPORTANT: describing changes in your final answer does NOT save them."
    + _EXCEL_CONTEXT
    + _DATA_SUFFICIENCY_POLICY
)
_UPDATER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets,
                   update_cell, replace_all_excel, batch_edit_excel, show_excel_history]

_DELETER_PROMPT = (
    "ROLE: Spreadsheet Deleter\n"
    "GOAL: Remove specific cells or rows from the spreadsheet safely.\n\n"
    "Deletion is permanent. Your workflow: 1) Search to confirm the right "
    "cell/row, 2) Delete using exact positions. For MORE THAN 2-3 "
    "deletions, use batch_edit_excel in one call. "
    "IMPORTANT: you are not done until the delete tool has been called."
    + _EXCEL_CONTEXT
)
_DELETER_TOOLS = [search_spreadsheet, vector_search_excel, list_sheets,
                   delete_cell, delete_row, batch_edit_excel, show_excel_history]

_FORMULA_PROMPT = (
    "ROLE: Formula Generator\n"
    "GOAL: Generate and insert correct Excel formulas. NEVER guess ranges.\n\n"
    "Your strict workflow:\n"
    "1) get_data_bounds to see exact row/column boundaries\n"
    "2) get_row_data with row=1 to read column headers\n"
    "3) get_column_data to inspect target column values\n"
    "4) Craft the formula using EXACT ranges from what you found\n"
    "5) insert_formula to place it in the right cell\n\n"
    "IMPORTANT: writing the formula in your answer does NOT insert it — "
    "you must call insert_formula."
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
# SECTION 8B: SPECIALIST SUBGRAPHS
# ================================================================

class SpecialistState(TypedDict):
    messages: Annotated[list, add_messages]


def _build_specialist_subgraph(node_name: str, system_prompt: str, tools: list, llm):
    """Build one specialist's ReAct loop as a compiled LangGraph subgraph."""
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
    """Compile the 5 specialist subgraphs."""
    return {
        name: _build_specialist_subgraph(name, prompt, tools, llm)
        for name, (prompt, tools) in _SPECIALIST_DEFS.items()
    }


# ================================================================
# SECTION 9: CREW GRAPH BUILDER
# ================================================================

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
    messages: Annotated[list, add_messages]
    pending: List[str]


def route_agents(user_request: str) -> List[str]:
    """Fast, code-based router — keyword check instead of LLM call."""
    text = user_request.lower()
    matched = [name for name, keywords in _INTENT_ROUTES
               if any(kw in text for kw in keywords)]
    if matched:
        return matched

    if _looks_like_write_request(user_request):
        return ["updater"]

    return ["reader"]


def _route_node(state: CrewState) -> CrewState:
    last_human = next(
        (m for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
        None,
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
    """Build the parent LangGraph that routes to specialist subgraphs."""
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
    """Extract plain text from an LLM message's content."""
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
    """Find the last AI message that has visible text."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            text = _extract_text(m.content).strip()
            if text:
                return text
    return "(no response)"


# ================================================================
# SECTION 10: MAIN SYSTEM
# ================================================================

def setup_llm():
    """Get a working LLM — load saved credentials or ask interactively."""
    if has_credentials():
        creds = load_credentials()
        info(f"Using saved credentials (model: {creds['model']})")
        return get_agentic_llm(creds["api_key"], creds["model"])

    header("First-time setup")

    while True:
        api_key = prompt_input("Gemini API key: ")
        if not api_key:
            warn("API key cannot be empty.\n")
            continue

        idx = choice_prompt("Select a model:", list(SUPPORTED_MODELS))
        model = SUPPORTED_MODELS[idx]

        spinner = Spinner(
            f"Testing connection to {model}...",
            done_label="Connected — credentials saved",
        ).start()
        result = validate_api_key(api_key, model)
        spinner.stop(ok=result["success"])

        if result["success"]:
            save_credentials(api_key, model)
            return get_agentic_llm(api_key, model)

        error(f"{result['error']}\n")


class ExcelAgentSystem:
    """Main orchestrator for Excel files.

    Takes a filepath, processes it, and runs a chatbot loop
    where a LangGraph crew handles spreadsheet operations.
    """

    def __init__(self, filepath: str, llm=None):
        self.filepath = filepath

        if llm:
            self.llm = llm
        else:
            cred = load_credentials()
            if not cred:
                raise RuntimeError(
                    "No saved Gemini credentials found. Run onboarding first "
                    "(or pass an explicit llm= to ExcelAgentSystem)."
                )
            info(f"Using saved credentials (model: {cred['model']})")
            self.llm = get_agentic_llm(cred["api_key"], cred["model"])

        header("DOXL AI — Excel Agent")
        process_file(filepath)

        self.specialists = build_specialists(self.llm)
        self.graph = build_crew_graph(self.specialists)

        header("Ready", "Reader · Writer · Updater · Deleter · Formula Gen")
        info("Ask in plain English, e.g. \"find rows where revenue > 1000\".")
        info("'sheets' lists sheet names, 'history' shows changes, 'quit' exits.")
        info("Full-sheet 'view' is disabled — sheets can hold lakhs of rows.\n")

    def process_request(self, user_input: str) -> str:
        """Send one request through the LangGraph crew and get the result."""
        mgr = _state.get("excel_manager")
        changes_before = len(mgr.changelog.changes) if mgr else 0

        result_state = self.graph.invoke(
            {"messages": [HumanMessage(content=user_input)], "pending": []},
            config={"recursion_limit": 150},
        )

        flush_task_queue()
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
                "was NOT modified. Try rephrasing more explicitly, or ask again."
            )
        return result_str

    def chat(self):
        """Interactive chatbot loop. Runs until 'quit'."""
        while True:
            try:
                user_input = prompt_input("\nYou: ")
            except (KeyboardInterrupt, EOFError):
                console.print()
                success("Goodbye!")
                break

            if not user_input:
                continue

            if user_input.lower() in ("quit", "exit", "q", "bye"):
                success("Goodbye!")
                break

            if user_input.lower() == "view":
                warn("'view' is disabled — a sheet can have lakhs of rows. "
                     "Try 'sheets' or search naturally.")
                continue

            if user_input.lower() == "sheets":
                console.print(list_sheets.invoke("all"))
                continue

            if user_input.lower() == "history":
                console.print(show_excel_history.invoke("all"))
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
                # Reset global progress bar on error to avoid stale state
                global _task_progress
                if _task_progress:
                    _task_progress.finish(saved=False)
                    _task_progress = None
                error(str(e) if str(e) != "None" else "Request failed — the agent could not complete the task.")
                info("Try rephrasing your request.")


# ================================================================
# SECTION 11: ENTRY POINT
# ================================================================

def main():
    """CLI entry point: python excel_agent.py <filepath.xlsx>"""
    if len(sys.argv) < 2:
        console.print("Usage: python excel_agent.py <filepath.xlsx>")
        sys.exit(1)

    filepath = sys.argv[1]
    llm = None
    system = ExcelAgentSystem(filepath, llm=llm)
    system.chat()


if __name__ == "__main__":
    main()
