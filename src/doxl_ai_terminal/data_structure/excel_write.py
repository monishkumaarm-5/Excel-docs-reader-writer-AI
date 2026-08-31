#excel_write.py
import os
from typing import Optional

from doxl_ai_terminal.data_structure.excel import ExcelData
from doxl_ai_terminal.data_structure.formatting import apply_excel_cell_format


def write_excel(excel_data: ExcelData, output_path: str, source_path: Optional[str] = None):
    """Write ExcelData back to an .xlsx file.

    This used to build a brand-new `openpyxl.Workbook()` from scratch on
    every save, which silently threw away everything the data model
    doesn't track: original formatting, column widths, merged cells,
    charts, images, and any formula not represented as plain cell text.

    Instead, this loads whichever file already has the "real" formatting
    -- `source_path` if given (used by save_as, which copies the live,
    already-edited file to a new path), else `output_path` if it already
    exists (the normal in-place sync case) -- and edits that workbook.
    Only cell values, and any explicitly-requested formatting, are
    touched; everything else about the workbook (styles on untouched
    cells, charts, merged ranges, etc.) is left exactly as it was.
    """
    from openpyxl import Workbook, load_workbook
    from openpyxl.utils import column_index_from_string

    load_path = source_path or (output_path if os.path.exists(output_path) else None)

    if load_path and os.path.exists(load_path):
        wb = load_workbook(load_path)
    else:
        wb = Workbook()
        wb.remove(wb.active)  # remove default sheet — nothing to preserve

    data_sheet_names = {sheet.sheet_name for sheet in excel_data.sheets}

    # Drop sheets the data model no longer knows about (e.g. deleted),
    # but never touch a sheet the data model simply hasn't loaded.
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
            row = int(cell.row)
            col = column_index_from_string(cell.column)
            keep[(row, col)] = cell

        # Clear any cell that used to hold a value but no longer appears
        # in the data model (a delete), within the sheet's pre-existing
        # bounds. Cells beyond those bounds are simply new writes below —
        # nothing to clear there.
        if ws.max_row and ws.max_column:
            for row in ws.iter_rows(min_row=1, max_row=ws.max_row, min_col=1, max_col=ws.max_column):
                for wcell in row:
                    if wcell.value is not None and (wcell.row, wcell.column) not in keep:
                        wcell.value = None

        for (row, col), cell in keep.items():
            wcell = ws.cell(row=row, column=col, value=cell.data_excel)
            apply_excel_cell_format(wcell, cell)

    wb.save(output_path)
    print(f"Saved: {output_path}")
