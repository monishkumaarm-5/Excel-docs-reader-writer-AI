#manger_tools
import os

from langchain_core.tools import tool

from doxl_ai_terminal.Frontier.ChangeLogManager_docs import LiveDocManager
from doxl_ai_terminal.Frontier.changelogmanager_excel import LiveExcelManager

# Shared manager instances
managers = {
    "doc": None,
    "excel": None,
}


def load_file(filepath: str):
    """Load a file into the right manager"""
    ext = os.path.splitext(filepath)[1].lower()

    if ext == ".docx":
        managers["doc"] = LiveDocManager(filepath)
    elif ext == ".xlsx":
        managers["excel"] = LiveExcelManager(filepath)
    else:
        print(f"Unsupported: {ext}")


# ---- Doc Tools ----

@tool
def doc_view() -> str:
    """View all lines in the loaded Word document"""
    if not managers["doc"]:
        return "No document loaded."
    return managers["doc"].view()

@tool
def doc_search(keyword: str) -> str:
    """Search for a keyword in the Word document"""
    if not managers["doc"]:
        return "No document loaded."
    results = managers["doc"].search(keyword)
    return "\n".join([f"P{r.paragraph} L{r.line}: {r.line_str}" for r in results]) or "No matches."

@tool
def doc_update_line(paragraph: int, line: int, new_text: str) -> str:
    """Update a line and auto-save to file"""
    if not managers["doc"]:
        return "No document loaded."
    return managers["doc"].update_line(paragraph, line, new_text)

@tool
def doc_add_line(paragraph: int, line: int, text: str) -> str:
    """Add a new line and auto-save to file"""
    if not managers["doc"]:
        return "No document loaded."
    return managers["doc"].add_line(paragraph, line, text)

@tool
def doc_delete_line(paragraph: int, line: int) -> str:
    """Delete a line and auto-save to file"""
    if not managers["doc"]:
        return "No document loaded."
    return managers["doc"].delete_line(paragraph, line)

@tool
def doc_replace_all(old_text: str, new_text: str) -> str:
    """Find and replace across all lines, auto-save to file"""
    if not managers["doc"]:
        return "No document loaded."
    return managers["doc"].replace_all(old_text, new_text)

@tool
def doc_history() -> str:
    """Show all changes made so far"""
    if not managers["doc"]:
        return "No document loaded."
    return managers["doc"].history()


# ---- Excel Tools ----

@tool
def excel_view(sheet_name: str = None) -> str:
    """View all cells in the Excel file. Optional: filter by sheet name."""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].view(sheet_name)

@tool
def excel_search(keyword: str) -> str:
    """Search for a keyword in Excel"""
    if not managers["excel"]:
        return "No Excel loaded."
    results = managers["excel"].search(keyword)
    return "\n".join([f"{r['sheet']}[{r['cell']}] = {r['value']}" for r in results]) or "No matches."

@tool
def excel_update_cell(sheet_name: str, row: str, column: str, new_value: str) -> str:
    """Update a cell and auto-save to file"""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].update_cell(sheet_name, row, column, new_value)

@tool
def excel_add_cell(sheet_name: str, row: str, column: str, value: str) -> str:
    """Add a new cell and auto-save to file"""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].add_cell(sheet_name, row, column, value)

@tool
def excel_add_row(sheet_name: str, row_data: dict) -> str:
    """Add a full row and auto-save. row_data = {'A': 'val1', 'B': 'val2'}"""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].add_row(sheet_name, row_data)

@tool
def excel_delete_cell(sheet_name: str, row: str, column: str) -> str:
    """Delete a cell and auto-save to file"""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].delete_cell(sheet_name, row, column)

@tool
def excel_delete_row(sheet_name: str, row: str) -> str:
    """Delete an entire row and auto-save"""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].delete_row(sheet_name, row)

@tool
def excel_replace_all(old_text: str, new_text: str) -> str:
    """Find and replace across all cells, auto-save"""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].replace_all(old_text, new_text)

@tool
def excel_history() -> str:
    """Show all changes made so far"""
    if not managers["excel"]:
        return "No Excel loaded."
    return managers["excel"].history()