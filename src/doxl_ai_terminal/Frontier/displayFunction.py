# displayFunction.py
"""Rich-powered display helpers for documents and spreadsheets."""

from doxl_ai_terminal.pipeline.terminal_ui import (
    console, header, divider, file_loaded_panel, make_table, print_table,
)


def display_doc(doc_data):
    """Render a Word document's content as a Rich table."""
    file_loaded_panel(doc_data.filename, [
        ("Paragraphs", str(doc_data.total_paragraphs)),
        ("Lines", str(doc_data.total_lines)),
    ])

    table = make_table(
        title="Document Content",
        columns=["Para", "Line", "Text"],
        rows=[
            [str(line.paragraph), str(line.line), line.line_str]
            for line in doc_data.lines
        ],
    )
    # First two columns are narrow identifiers
    table.columns[0].style = "muted"
    table.columns[0].max_width = 6
    table.columns[1].style = "muted"
    table.columns[1].max_width = 6
    table.columns[2].style = ""
    console.print(table)


def display_excel(excel_data):
    """Render an Excel workbook's content as Rich tables — one per sheet."""
    file_loaded_panel(excel_data.filename, [
        ("Sheets", str(excel_data.total_sheets)),
    ])

    for sheet in excel_data.sheets:
        # Build a set of unique columns and rows in the sheet
        col_set = sorted({c.column for c in sheet.cells})
        row_set = sorted({c.row for c in sheet.cells})

        if not col_set or not row_set:
            header(sheet.sheet_name, "empty sheet")
            continue

        # Build a lookup map for fast cell access
        cell_map = {}
        for c in sheet.cells:
            cell_map[(c.row, c.column)] = str(c.data_excel) if c.data_excel is not None else ""

        # Build rows for the table
        table_rows = []
        for r in row_set:
            row_vals = [str(r)] + [cell_map.get((r, col), "") for col in col_set]
            table_rows.append(row_vals)

        subtitle = f"{sheet.total_rows} rows, {sheet.total_columns} columns"
        table = make_table(
            title=f"{sheet.sheet_name}  ({subtitle})",
            columns=["Row"] + col_set,
            rows=table_rows,
        )
        # Row-number column is a muted identifier
        table.columns[0].style = "muted"
        table.columns[0].max_width = 8
        console.print(table)
