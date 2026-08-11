import os

from src.doxl_ai_terminal.data_structure.docs import DocData, DocLine
from src.doxl_ai_terminal.data_structure.excel import ExcelData, ExcelSheet, ExcelCell


def read_word(filepath) -> DocData:
    from docx import Document

    doc = Document(filepath)
    doc_data = DocData(
        filename=os.path.basename(filepath),
        total_paragraphs=0,
        total_lines=0
    )

    for p_index, para in enumerate(doc.paragraphs, start=1):
        if para.text.strip() == "":
            continue

        for l_index, line in enumerate(para.text.split("\n"), start=1):
            if line.strip() == "":
                continue
            doc_data.lines.append(DocLine(
                paragraph=p_index,
                line=l_index,
                line_str=line.strip()
            ))
            doc_data.total_lines += 1

        doc_data.total_paragraphs += 1

    return doc_data


def read_excel(filepath) -> ExcelData:
    from openpyxl import load_workbook
    from openpyxl.utils import get_column_letter

    wb = load_workbook(filepath, read_only=True)
    excel_data = ExcelData(
        filename=os.path.basename(filepath),
        total_sheets=len(wb.sheetnames)
    )

    for sheet in wb:
        sheet_data = ExcelSheet(
            sheet_name=sheet.title,
            total_rows=0,
            total_columns=0
        )

        for row in sheet.iter_rows():
            for cell in row:
                if cell.value is not None:
                    sheet_data.cells.append(ExcelCell(
                        row=str(cell.row),
                        column=get_column_letter(cell.column),
                        data_excel=str(cell.value)
                    ))
                    sheet_data.total_rows = max(sheet_data.total_rows, cell.row)
                    sheet_data.total_columns = max(sheet_data.total_columns, cell.column)

        excel_data.sheets.append(sheet_data)

    wb.close()
    return excel_data