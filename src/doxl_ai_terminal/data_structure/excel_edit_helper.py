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


# ---- Update a cell ----
def update_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str, new_value: str) -> str:
    """Update a cell value in an Excel sheet by sheet name, row, and column."""
    cell = find_excel_cell(excel_data, sheet_name, row, column)
    if cell:
        old = cell.data_excel
        cell.data_excel = new_value
        return f"Updated: {sheet_name}[{column}{row}] '{old}' → '{new_value}'"
    return f"Not found: {sheet_name}[{column}{row}]"


# ---- Add a cell ----
def add_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str, value: str) -> str:
    """Add a new cell to an Excel sheet."""
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            sheet.cells.append(ExcelCell(row=row, column=column, data_excel=value))
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