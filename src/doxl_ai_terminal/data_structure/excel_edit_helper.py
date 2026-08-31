#excel_edit_helper.py
# ---- Find a cell ----

from typing import List, Optional
from doxl_ai_terminal.data_structure.excel import ExcelData, ExcelCell


def find_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str) -> Optional[ExcelCell]:
    """Internal helper — find a cell by sheet, row, and column."""
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            for cell in sheet.cells:
                if cell.row == row and cell.column == column:
                    return cell
    return None


_FORMAT_FIELDS = (
    "font_name", "font_size", "bold", "italic", "underline",
    "font_color", "fill_color", "alignment", "number_format",
)


def _apply_format_kwargs(cell: ExcelCell, format_kwargs: dict) -> dict:
    applied = {}
    for key, val in format_kwargs.items():
        if key in _FORMAT_FIELDS and val is not None:
            setattr(cell, key, val)
            applied[key] = val
    return applied


# ---- Update a cell ----
def update_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str, new_value: str, **format_kwargs) -> str:
    """Update a cell value (and optionally its formatting) in an Excel
    sheet by sheet name, row, and column."""
    cell = find_excel_cell(excel_data, sheet_name, row, column)
    if cell:
        old = cell.data_excel
        cell.data_excel = new_value
        _apply_format_kwargs(cell, format_kwargs)
        return f"Updated: {sheet_name}[{column}{row}] '{old}' → '{new_value}'"
    return f"Not found: {sheet_name}[{column}{row}]"


# ---- Add a cell ----
def add_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str, value: str, **format_kwargs) -> str:
    """Add a new cell to an Excel sheet. If a cell already exists at this
    (row, column), it is updated in place instead of appended a second
    time — a duplicate ExcelCell at the same address would show up twice
    in search results even though only the last one written survives a
    save to disk."""
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            existing = find_excel_cell(excel_data, sheet_name, row, column)
            if existing:
                old = existing.data_excel
                existing.data_excel = value
                _apply_format_kwargs(existing, format_kwargs)
                return f"Updated (cell already had a value): {sheet_name}[{column}{row}] '{old}' -> '{value}'"

            new_cell = ExcelCell(row=row, column=column, data_excel=value)
            _apply_format_kwargs(new_cell, format_kwargs)
            sheet.cells.append(new_cell)
            return f"Added: {sheet_name}[{column}{row}] = '{value}'"
    return f"Sheet not found: {sheet_name}"


# ---- Delete a cell ----
def delete_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str) -> str:
    """Delete a cell from an Excel sheet by sheet name, row, and column."""
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            cell = find_excel_cell(excel_data, sheet_name, row, column)
            if cell:
                sheet.cells.remove(cell)
                return f"Deleted: {sheet_name}[{column}{row}]"
    return f"Not found: {sheet_name}[{column}{row}]"


# ---- Search cells ----
def search_excel(excel_data: ExcelData, keyword: str) -> List[ExcelCell]:
    """Search for a keyword across all cells in the Excel workbook."""
    results = []
    for sheet in excel_data.sheets:
        for cell in sheet.cells:
            if keyword.lower() in cell.data_excel.lower():
                results.append(cell)
    return results


# ---- Replace text in all cells ----
def replace_excel_text(excel_data: ExcelData, old_text: str, new_text: str) -> str:
    """Find and replace text across all cells in the Excel workbook."""
    count = 0
    for sheet in excel_data.sheets:
        for cell in sheet.cells:
            if old_text in cell.data_excel:
                cell.data_excel = cell.data_excel.replace(old_text, new_text)
                count += 1
    return f"Replaced '{old_text}' → '{new_text}' in {count} cells"