from doxl_ai_terminal.pipeline.terminal_ui import header, divider


def display_doc(doc_data):
    header(doc_data.filename, f"{doc_data.total_paragraphs} paragraphs · {doc_data.total_lines} lines")

    for line in doc_data.lines:
        print(f"  P{line.paragraph} | L{line.line} | {line.line_str}")


def display_excel(excel_data):
    header(excel_data.filename, f"{excel_data.total_sheets} sheet(s)")

    for sheet in excel_data.sheets:
        print(f"\n  {sheet.sheet_name}  ({sheet.total_rows} rows, {sheet.total_columns} columns)")
        divider()

        for cell in sheet.cells:
            print(f"    [{cell.column}{cell.row}] = {cell.data_excel}")