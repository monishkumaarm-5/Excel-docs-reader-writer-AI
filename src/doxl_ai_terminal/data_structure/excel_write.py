from doxl_ai_terminal.data_structure.excel import ExcelData
from langchain_core.tools import tool

@tool
def write_excel(excel_data: ExcelData, output_path: str):
    from openpyxl import Workbook
    from openpyxl.utils import column_index_from_string

    wb = Workbook()
    wb.remove(wb.active)  # remove default sheet

    for sheet_data in excel_data.sheets:
        ws = wb.create_sheet(title=sheet_data.sheet_name)

        for cell in sheet_data.cells:
            row = int(cell.row)
            col = column_index_from_string(cell.column)
            ws.cell(row=row, column=col, value=cell.data_excel)

    wb.save(output_path)
    print(f"Saved: {output_path}")