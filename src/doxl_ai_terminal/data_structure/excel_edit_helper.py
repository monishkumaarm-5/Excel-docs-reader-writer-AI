# ---- Find a cell ----
from typing import List
from langchain_core.tools import tool
from doxl_ai_terminal.data_structure.excel import ExcelData, ExcelCell

@tool
def find_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str) -> ExcelCell:
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            for cell in sheet.cells:
                if cell.row == row and cell.column == column:
                    return cell
    return None


# ---- Update a cell ----
@tool
def update_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str, new_value: str):
    cell = find_excel_cell(excel_data, sheet_name, row, column)
    if cell:
        old = cell.data_excel
        cell.data_excel = new_value
        print(f"Updated: {sheet_name}[{column}{row}] '{old}' → '{new_value}'")
    else:
        print(f"Not found: {sheet_name}[{column}{row}]")


# ---- Add a cell ----
@tool
def add_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str, value: str):
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            sheet.cells.append(ExcelCell(row=row, column=column, data_excel=value))
            print(f"Added: {sheet_name}[{column}{row}] = '{value}'")
            return
    print(f"Sheet not found: {sheet_name}")


# ---- Delete a cell ----
@tool
def delete_excel_cell(excel_data: ExcelData, sheet_name: str, row: str, column: str):
    for sheet in excel_data.sheets:
        if sheet.sheet_name == sheet_name:
            cell = find_excel_cell(excel_data, sheet_name, row, column)
            if cell:
                sheet.cells.remove(cell)
                print(f"Deleted: {sheet_name}[{column}{row}]")
                return
    print(f"Not found: {sheet_name}[{column}{row}]")


# ---- Search cells ----
@tool
def search_excel(excel_data: ExcelData, keyword: str) -> List[ExcelCell]:
    results = []
    for sheet in excel_data.sheets:
        for cell in sheet.cells:
            if keyword.lower() in cell.data_excel.lower():
                results.append(cell)
    return results


# ---- Replace text in all cells ----
@tool
def replace_excel_text(excel_data: ExcelData, old_text: str, new_text: str):
    count = 0
    for sheet in excel_data.sheets:
        for cell in sheet.cells:
            if old_text in cell.data_excel:
                cell.data_excel = cell.data_excel.replace(old_text, new_text)
                count += 1
    print(f"Replaced '{old_text}' → '{new_text}' in {count} cells")