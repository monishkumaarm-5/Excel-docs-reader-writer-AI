
def display_doc(doc_data):
    print(f"\nFile: {doc_data.filename}")
    print(f"Paragraphs: {doc_data.total_paragraphs}")
    print(f"Total Lines: {doc_data.total_lines}")
    print("-" * 50)

    for line in doc_data.lines:
        print(f"  P{line.paragraph} | L{line.line} | {line.line_str}")


def display_excel(excel_data):
    print(f"\nFile: {excel_data.filename}")
    print(f"Total Sheets: {excel_data.total_sheets}")
    print("-" * 50)

    for sheet in excel_data.sheets:
        print(f"\n  Sheet: {sheet.sheet_name}")
        print(f"  Rows: {sheet.total_rows} | Columns: {sheet.total_columns}")
        print()

        for cell in sheet.cells:
            print(f"    [{cell.column}{cell.row}] = {cell.data_excel}")