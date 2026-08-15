# excel_agent.py
"""
CrewAI Multi-Agent Excel Handler
==================================

Architecture:
    Manager Agent (routes tasks)
        ├── Reader Agent        → list sheets, exact search, RAG semantic search
        ├── Writer Agent        → add cells, add rows
        ├── Updater Agent       → update cells, find-replace
        ├── Deleter Agent       → delete cells, delete rows
        └── Formula Generator   → inspect data, craft & insert Excel formulas

Flow:
    1. User gives filepath → file is read, chunked, vectorized
    2. Chatbot loop starts
    3. User types a request
    4. CrewAI Manager delegates to the right specialist
    5. Specialist uses tools → result returned
    6. Loop continues until user quits

Usage:
    python excel_agent.py sales_data.xlsx
"""

import os
import sys
from typing import Optional, List, Dict

from crewai import Agent, Task, Crew, Process
from crewai.tools import tool as crewai_tool

# ---- Your existing modules ----
from doxl_ai_terminal.Frontier.changelogmanager_excel import LiveExcelManager
from doxl_ai_terminal.Frontier.fileReader import read_excel
from doxl_ai_terminal.Chunker.chunking import chunk_excel_row_wise, chunk_excel_column_wise
from doxl_ai_terminal.data_handler.vector_config import store_in_chroma, make_collection_name
from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager
from doxl_ai_terminal.pipeline.config import SUPPORTED_MODELS, validate_api_key, get_crewai_llm
from doxl_ai_terminal.pipeline.credential_store import (
    has_credentials,
    load_credentials,
    save_credentials,
)
# ================================================================
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

    # ---- Step 1: Read the spreadsheet into ExcelData ----
    print(f"\n  [1/4] Reading: {filepath}")
    excel_data = read_excel(filepath)
    _state["excel_data"] = excel_data

    sheet_names = [s.sheet_name for s in excel_data.sheets]
    total_cells = sum(len(s.cells) for s in excel_data.sheets)
    print(f"         {len(sheet_names)} sheet(s): {', '.join(sheet_names)}")
    print(f"         {total_cells} cells total")

    # ---- Step 2: Load LiveExcelManager for editing ----
    print(f"  [2/4] Loading editor...")
    _state["excel_manager"] = LiveExcelManager(filepath)

    # ---- Steps 3-4: Chunk + store in vector DB + load for search ----
    print(f"  [3/4] Chunking and vectorizing...")
    _rebuild_excel_vector_index(filepath, excel_data)
    print(f"  [4/4] Vector search ready ({len(_state['vector_managers'])} collection(s)).")

    print(f"\n  Ready! {total_cells} cells across {len(sheet_names)} sheet(s).\n")


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
        print(f"         {len(chunks)} chunks -> '{collection_name}'")

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
        print(f"  [warn] Refreshing sheet data failed: {e}")


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
        print(f"  [warn] Vector index refresh failed: {e}")


# ================================================================
# SECTION 3: TOOLS — READER
# ================================================================
# These tools let the Reader Agent search the spreadsheet. There is
# deliberately NO "view everything" tool here — see the note below.

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
# @crewai_tool
# def view_spreadsheet(sheet_name: str = "") -> str:
#     """View all cells in the loaded Excel file.
#     Pass a sheet name to filter by one sheet, or leave empty to see all.
#     Shows sheet name, cell position [ColumnRow], and value."""
#     mgr = _state["excel_manager"]
#     if not mgr:
#         return "ERROR: No spreadsheet loaded."
#     return mgr.view(sheet_name if sheet_name else None)
# ─────────────────────────────────────────────────────────────────


@crewai_tool
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


@crewai_tool
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


@crewai_tool
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

@crewai_tool
def add_cell(sheet_name: str, row: str, column: str, value: str) -> str:
    """Add a new cell to the spreadsheet.
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


@crewai_tool
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

@crewai_tool
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


@crewai_tool
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


# ================================================================
# SECTION 6: TOOLS — DELETER
# ================================================================

@crewai_tool
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


@crewai_tool
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
# These tools let the Formula Agent inspect data and insert formulas.

@crewai_tool
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


@crewai_tool
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


@crewai_tool
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


@crewai_tool
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
# SECTION 7: TOOLS — SHARED (all agents can use)
# ================================================================

@crewai_tool
def show_excel_history(check: str = "all") -> str:
    """Show all changes made to the spreadsheet so far.
    Displays: what was changed, old value, new value, and when."""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    return mgr.history()


@crewai_tool
def save_excel_copy(output_path: str) -> str:
    """Save the spreadsheet to a NEW file path (keeps original unchanged).
    Args:
        output_path: full path for the new file (e.g., 'sales_v2.xlsx')"""
    mgr = _state["excel_manager"]
    if not mgr:
        return "ERROR: No spreadsheet loaded."
    return mgr.save_as(output_path)


@crewai_tool
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
# SECTION 8: AGENT DEFINITIONS
# ================================================================
# Each agent gets ONLY the tools it needs.
# Excel agents understand sheets, rows, columns, cells.

def build_agents(llm=None):
    """
    Create the 5 specialist agents + 1 manager agent.

    Hierarchy:
        Manager (routes + oversees)
            ├── Reader        (view, search, vector search, list sheets)
            ├── Writer        (add cells, add rows)
            ├── Updater       (update cells, replace all)
            ├── Deleter       (delete cells, delete rows)
            └── Formula Gen   (inspect data, generate & insert formulas)
    """
    llm_config = {"llm": llm} if llm else {}

    # ---- READER AGENT ----
    reader = Agent(
        role="Spreadsheet Reader",
        goal=(
            "Read and search spreadsheet content accurately. "
            "Use keyword search for exact cell matches, vector search for "
            "meaning-based queries across rows and columns."
        ),
        backstory=(
            "You are the data analyst. You know every sheet, row, and column. "
            "Your workflow: first list_sheets to see what exists, then "
            "search_spreadsheet for exact matches or vector_search_excel "
            "for conceptual/meaning-based queries — there is no 'view "
            "everything' tool, since a large sheet can have lakhs of rows "
            "and would never fit in context. Always search instead of "
            "trying to see the whole sheet at once. "
            "Excel data is organized as Sheet -> Row -> Column -> Cell."
        ),
        tools=[
            search_spreadsheet,
            vector_search_excel,
            list_sheets,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- WRITER AGENT ----
    writer = Agent(
        role="Spreadsheet Writer",
        goal=(
            "Add new data to the spreadsheet at the correct position. "
            "Always check the sheet structure first before adding."
        ),
        backstory=(
            "You are the data entry specialist. Before adding anything, you "
            "ALWAYS use list_sheets to see sheet names, then search_spreadsheet "
            "or vector_search_excel to understand the structure (headers in "
            "row 1, data below) — you never try to view the whole sheet, "
            "since it may hold lakhs of rows. "
            "You can add individual cells or entire rows. "
            "Columns use letters (A, B, C...), rows use numbers (1, 2, 3...)."
        ),
        tools=[
            search_spreadsheet,
            vector_search_excel,
            list_sheets,
            add_cell,
            add_row,
            show_excel_history,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- UPDATER AGENT ----
    updater = Agent(
        role="Spreadsheet Updater",
        goal=(
            "Modify existing cell values accurately. "
            "Find the exact cell first, then update it."
        ),
        backstory=(
            "You are the data editor. You never update blindly. Your workflow: "
            "1) Search for the cell to change, 2) Note the exact sheet name, "
            "row number, and column letter, 3) Update with the new value, "
            "4) Verify the change. For bulk changes, use replace_all_excel. "
            "Cell positions look like: SheetName[ColumnRow] e.g., Sheet1[B3]."
        ),
        tools=[
            search_spreadsheet,
            vector_search_excel,
            list_sheets,
            update_cell,
            replace_all_excel,
            show_excel_history,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- DELETER AGENT ----
    deleter = Agent(
        role="Spreadsheet Deleter",
        goal=(
            "Remove specific cells or rows from the spreadsheet safely. "
            "Always confirm the exact position before deleting."
        ),
        backstory=(
            "You are the data cleanup specialist. Deletion is permanent, "
            "so you are extra careful. Your workflow: 1) Search for the "
            "content to delete with search_spreadsheet or vector_search_excel "
            "to confirm it's the right cell/row (never try to view the "
            "whole sheet — it may hold lakhs of rows), 2) Delete using the "
            "exact sheet, row, column. "
            "You can delete a single cell or an entire row."
        ),
        tools=[
            search_spreadsheet,
            vector_search_excel,
            list_sheets,
            delete_cell,
            delete_row,
            show_excel_history,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- FORMULA GENERATOR AGENT ----
    formula_gen = Agent(
        role="Formula Generator",
        goal=(
            "Generate and insert correct Excel formulas into the spreadsheet. "
            "Understand the data layout first, then craft the precise formula. "
            "Support all Excel functions: SUM, AVERAGE, COUNT, COUNTIF, SUMIF, "
            "IF, VLOOKUP, INDEX, MATCH, MIN, MAX, CONCATENATE, and more."
        ),
        backstory=(
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
            "For VLOOKUP, always check both the lookup sheet and data sheet first."
        ),
        tools=[
            list_sheets,
            search_spreadsheet,
            get_data_bounds,
            get_column_data,
            get_row_data,
            insert_formula,
            show_excel_history,
        ],
        allow_delegation=False,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    # ---- MANAGER AGENT ----
    manager = Agent(
        role="Spreadsheet Operations Manager",
        goal=(
            "Understand the user's request and delegate to the right agent. "
            "READ/SEARCH/VIEW requests -> Reader Agent. "
            "ADD/INSERT/WRITE requests -> Writer Agent. "
            "UPDATE/EDIT/MODIFY/REPLACE requests -> Updater Agent. "
            "DELETE/REMOVE requests -> Deleter Agent. "
            "FORMULA/SUM/AVERAGE/COUNT/CALCULATE/TOTAL/VLOOKUP requests -> Formula Generator. "
            "Review results before returning to the user."
        ),
        backstory=(
            "You are the team lead for spreadsheet operations. You receive "
            "user requests and figure out which specialist should handle it. "
            "Sometimes a task needs multiple steps (e.g., 'find all sales "
            "below 100 and delete them' needs the Reader first, then the "
            "Deleter). Any request involving formulas, calculations, totals, "
            "sums, averages, or Excel functions goes to the Formula Generator. "
            "You coordinate the work and ensure accuracy."
        ),
        allow_delegation=True,
        verbose=True,
        max_iter=15,
        **llm_config,
    )

    return {
        "reader": reader,
        "writer": writer,
        "updater": updater,
        "deleter": deleter,
        "formula_gen": formula_gen,
        "manager": manager,
    }


# ================================================================
# SECTION 9: CREW BUILDER
# ================================================================

def build_crew(agents: dict, user_request: str):
    """
    Build a CrewAI Crew for one user request.

    - Process: HIERARCHICAL (manager routes to specialists)
    - Memory: DISABLED — CrewAI's built-in memory needs its own embedder,
      and without one explicitly configured it defaults to trying an
      OpenAI embedding call. Since this project has no OpenAI key (and
      is meant to run fully offline on the local sentence-transformers
      model — see vector_config.py), that default silently stalls/retries
      on every single kickoff, which is a big chunk of the "everything
      is slow" complaint. We already have our own persistent, local,
      offline RAG layer (Chroma) for file content, so CrewAI's separate
      conversational memory isn't needed on top of it.
    - Verbose: DISABLED at the crew level too — same reasoning as the
      agents: writing full reasoning traces to the console on every
      step adds real wall-clock time over a long multi-agent run.
    """
    task = Task(
        description=(
            f"USER REQUEST:\n"
            f"{user_request}\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Determine what the user wants (read/search/add/update/delete/formula)\n"
            f"2. Delegate to the right specialist agent\n"
            f"3. The specialist should use tools to complete the task\n"
            f"4. Return a clear, concise summary of what was done or found\n\n"
            f"ROUTING:\n"
            f"- Read/search/view → Reader\n"
            f"- Add/insert/write → Writer\n"
            f"- Update/edit/modify → Updater\n"
            f"- Delete/remove → Deleter\n"
            f"- Formula/sum/average/count/calculate/total/vlookup → Formula Generator\n\n"
            f"EXCEL CONTEXT:\n"
            f"- Data is in sheets. Use list_sheets to see them.\n"
            f"- Cells are addressed as SheetName[ColumnRow], e.g., Sheet1[B3]\n"
            f"- Row 1 usually has headers. Data starts from row 2.\n"
            f"- Columns use letters (A, B, C...), rows use numbers (1, 2, 3...)."
        ),
        expected_output=(
            "A clear summary of the action taken and its result. "
            "If data was found, show it in a readable format. "
            "If data was changed, show what was changed (old → new). "
            "If a formula was inserted, show the formula and its cell position."
        ),
    )

    crew = Crew(
        agents=[
            agents["reader"],
            agents["writer"],
            agents["updater"],
            agents["deleter"],
            agents["formula_gen"],
        ],
        tasks=[task],
        process=Process.hierarchical,
        manager_agent=agents["manager"],
        memory=False,
        verbose=True,
    )

    return crew


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
        print(f"  Using saved credentials (model: {creds['model']})")
        return get_crewai_llm(creds["api_key"], creds["model"])

    # ---- First run: ask the user ----
    print("\n  First-time setup")
    print("  " + "-" * 40)

    while True:
        api_key = input("  Gemini API key: ").strip()
        if not api_key:
            print("  API key cannot be empty.\n")
            continue

        print("\n  Available models:")
        for i, m in enumerate(SUPPORTED_MODELS, 1):
            print(f"    {i}. {m}")

        choice = input(f"\n  Pick a model (1-{len(SUPPORTED_MODELS)}): ").strip()
        try:
            model = SUPPORTED_MODELS[int(choice) - 1]
        except (ValueError, IndexError):
            print("  Invalid choice.\n")
            continue

        # ---- Test before saving ----
        print(f"\n  Testing connection to {model}...")
        result = validate_api_key(api_key, model)

        if result["success"]:
            save_credentials(api_key, model)
            print("  Connected. Credentials saved.\n")
            return get_crewai_llm(api_key, model)

        print(f"  {result['error']}\n")
class ExcelAgentSystem:
    """
    The main orchestrator for Excel files.

    Takes a filepath, processes it, and runs a chatbot loop
    where CrewAI agents handle spreadsheet operations.

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
            llm: (optional) LangChain LLM instance for the agents.
                 Defaults to Gemini 2.0 Flash (using your .env key).
                 You can pass any LangChain-compatible LLM.
        """
        self.filepath = filepath

        # ---- Set up the LLM ----
        if llm:
            self.llm = llm
        else:
            # Default: load saved Gemini credentials and build a
            # CrewAI-native LLM (provider-prefixed model string).
            cred = load_credentials()
            if not cred:
                raise RuntimeError(
                    "No saved Gemini credentials found. Run onboarding first "
                    "(or pass an explicit llm= to ExcelAgentSystem)."
                )
            print(f"Using CrewAI Gemini LLM with model: {cred['model']}")
            self.llm = get_crewai_llm(cred["api_key"], cred["model"])

        # ---- Step 1: Process the file ----
        print("=" * 55)
        print("  DOXL AI — Excel Agent System")
        print("=" * 55)
        process_file(filepath)

        # ---- Step 2: Build agents ----
        self.agents = build_agents(self.llm)

        print("=" * 55)
        print("  Agents Ready:")
        print("    - Reader      (exact + RAG semantic search)")
        print("    - Writer      (add cells & rows)")
        print("    - Updater     (modify cells)")
        print("    - Deleter     (remove cells & rows)")
        print("    - Formula Gen (generate & insert formulas)")
        print("    - Manager     (routes your requests)")
        print("=" * 55)
        print("  Commands:")
        print("    Type your request in plain English (e.g. 'find rows where")
        print("    revenue > 1000' or 'search for John') — full-sheet 'view'")
        print("    is disabled since sheets can hold lakhs of rows.")
        print("    'sheets'  — list all sheet names")
        print("    'history' — see all changes made")
        print("    'quit'    — exit the system")
        print("=" * 55)

    def process_request(self, user_input: str) -> str:
        """
        Send one request through CrewAI and get the result.

        The Manager agent reads the request, picks the right
        specialist, and returns the result. If anything in this
        request wrote to the sheet, the (expensive) vector index is
        rebuilt exactly once here — after the whole request finishes —
        rather than after every individual tool call.
        """
        crew = build_crew(self.agents, user_input)
        result = crew.kickoff()
        rebuild_excel_vector_index_if_dirty()
        return str(result)

    def chat(self):
        """
        Interactive chatbot loop.

        Keeps running until the user types 'quit'.
        Each request goes through CrewAI agents.
        Memory is preserved across requests in the same session.
        """
        while True:
            # ---- Get user input ----
            try:
                user_input = input("\nYou: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nGoodbye!")
                break

            if not user_input:
                continue

            # ---- Check for exit ----
            if user_input.lower() in ("quit", "exit", "q", "bye"):
                print("Goodbye!")
                break

            # ---- Quick shortcuts (skip CrewAI for speed) ----
            if user_input.lower() == "view":
                print(
                    "\n  'view' is disabled — a sheet can have lakhs of rows, "
                    "which won't fit in context.\n"
                    "  Try 'sheets' to see what's there, or search naturally, "
                    "e.g. 'find rows with status = pending'."
                )
                continue

            if user_input.lower() == "sheets":
                print("\n" + list_sheets.invoke("all"))
                continue

            if user_input.lower() == "history":
                print("\n" + show_excel_history.invoke("all"))
                continue

            # ---- Process through CrewAI ----
            try:
                print("\n  Processing your request...\n")
                result = self.process_request(user_input)
                print(f"\nAgent: {result}")

            except Exception as e:
                print(f"\n  Error: {e}")
                print("  Try rephrasing your request.")


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
    # Option 1: Gemini (default — uses GEMINI_API_KEY from your .env)
    #   llm = None  → auto-loads gemini-2.0-flash
    #
    # Option 2: Gemini Pro (heavier, more accurate)
    #   from langchain_google_genai import ChatGoogleGenerativeAI
    #   from doxl_ai_terminal.config import GEMINI_API_KEY
    #   llm = ChatGoogleGenerativeAI(
    #       model="gemini-2.5-pro",
    #       google_api_key=GEMINI_API_KEY,
    #       temperature=0,
    #   )
    #
    # Option 3: Ollama (fully local, no API key)
    #   from langchain_ollama import ChatOllama
    #   llm = ChatOllama(model="llama3.1", temperature=0)
    #
    llm=None
   # None = uses default (Gemini 2.0 Flash)

    system = ExcelAgentSystem(filepath, llm=llm)
    system.chat()


if __name__ == "__main__":
    main()