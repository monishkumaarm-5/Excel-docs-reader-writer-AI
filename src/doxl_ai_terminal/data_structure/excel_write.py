#excel_write.py
from doxl_ai_terminal.data_structure.excel import ExcelData


def write_excel(excel_data: ExcelData, output_path: str):
    """Write ExcelData back to an .xlsx file."""
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