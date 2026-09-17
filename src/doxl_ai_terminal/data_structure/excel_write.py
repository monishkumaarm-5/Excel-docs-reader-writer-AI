# excel_write.py
"""Write ExcelData back to an .xlsx file."""

import os
from typing import Optional

from doxl_ai_terminal.data_structure.excel import ExcelData, coerce_excel_value
from doxl_ai_terminal.data_structure.formatting import apply_excel_cell_format


def write_excel(excel_data: ExcelData, output_path: str, source_path: Optional[str] = None):
    """Write ExcelData back to an .xlsx file.

    Loads the existing workbook (from source_path or output_path) and
    edits it in place, preserving formatting, charts, merged cells, etc.
    Only cell values and explicitly-requested formatting are touched.
    """
    from openpyxl import Workbook, load_workbook
    from openpyxl.utils import column_index_from_string

    load_path = source_path or (output_path if os.path.exists(output_path) else None)

    if load_path and os.path.exists(load_path):
        wb = load_workbook(load_path)
    else:
        wb = Workbook()
        wb.remove(wb.active)

    data_sheet_names = {sheet.sheet_name for sheet in excel_data.sheets}

    for name in list(wb.sheetnames):
        if name not in data_sheet_names:
            del wb[name]

    for sheet_data in excel_data.sheets:
        if sheet_data.sheet_name in wb.sheetnames:
            ws = wb[sheet_data.sheet_name]
        else:
            ws = wb.create_sheet(title=sheet_data.sheet_name)

        keep = {}
        for cell in sheet_data.cells:
            # cell.row is already int — no conversion needed
            col = column_index_from_string(cell.column)
            keep[(cell.row, col)] = cell

        # Clear cells that were deleted from the data model
        if ws.max_row and ws.max_column:
            for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
                for wcell in row:
                    if wcell.value is not None and (wcell.row, wcell.column) not in keep:
                        wcell.value = None

        for (row, col), cell in keep.items():
            value = coerce_excel_value(cell.data_excel)
            wcell = ws.cell(row=row, column=col, value=value)
            apply_excel_cell_format(wcell, cell)

    wb.save(output_path)
