# ---- Doc: Line by Line ----
from typing import Dict, List

from src.doxl_ai_terminal.data_structure.excel import Chunk, DocData, ExcelData


def chunk_doc_line_by_line(doc_data: DocData) -> List[Chunk]:
    chunks = []
    for line in doc_data.lines:
        chunks.append(Chunk(
            chunk_id=f"{doc_data.filename}_p{line.paragraph}_l{line.line}",
            text=line.line_str,
            metadata={
                "source": doc_data.filename,
                "type": "doc",
                "format": "line_by_line",
                "paragraph": line.paragraph,
                "line": line.line,
            }
        ))
    return chunks


# ---- Doc: Sub-Para by Sub-Para ----

def chunk_doc_sub_para(doc_data: DocData) -> List[Chunk]:
    chunks = []
    para_map: Dict[int, List[str]] = {}

    for line in doc_data.lines:
        if line.paragraph not in para_map:
            para_map[line.paragraph] = []
        para_map[line.paragraph].append(line.line_str)

    for p_num, lines in para_map.items():
        chunks.append(Chunk(
            chunk_id=f"{doc_data.filename}_p{p_num}",
            text=" ".join(lines),
            metadata={
                "source": doc_data.filename,
                "type": "doc",
                "format": "sub_para",
                "paragraph": p_num,
                "total_lines": len(lines),
            }
        ))
    return chunks


# ---- Excel: Row Wise ----

def chunk_excel_row_wise(excel_data: ExcelData) -> List[Chunk]:
    chunks = []
    for sheet in excel_data.sheets:
        headers = {}
        row_data: Dict[str, List] = {}

        for cell in sheet.cells:
            if cell.row == "1":
                headers[cell.column] = cell.data_excel
            else:
                if cell.row not in row_data:
                    row_data[cell.row] = []
                row_data[cell.row].append((cell.column, cell.data_excel))

        for row_num, pairs in row_data.items():
            text_parts = [f"{headers.get(col, col)}: {val}" for col, val in pairs]
            chunks.append(Chunk(
                chunk_id=f"{excel_data.filename}_{sheet.sheet_name}_row{row_num}",
                text=" | ".join(text_parts),
                metadata={
                    "source": excel_data.filename,
                    "type": "excel",
                    "format": "row_wise",
                    "sheet": sheet.sheet_name,
                    "row": row_num,
                }
            ))
    return chunks


# ---- Excel: Column Wise ----

def chunk_excel_column_wise(excel_data: ExcelData) -> List[Chunk]:
    chunks = []
    for sheet in excel_data.sheets:
        headers = {}
        col_data: Dict[str, List[str]] = {}

        for cell in sheet.cells:
            if cell.row == "1":
                headers[cell.column] = cell.data_excel
            else:
                if cell.column not in col_data:
                    col_data[cell.column] = []
                col_data[cell.column].append(cell.data_excel)

        for col_letter, values in col_data.items():
            header = headers.get(col_letter, col_letter)
            chunks.append(Chunk(
                chunk_id=f"{excel_data.filename}_{sheet.sheet_name}_col{col_letter}",
                text=f"{header}: " + ", ".join(values),
                metadata={
                    "source": excel_data.filename,
                    "type": "excel",
                    "format": "column_wise",
                    "sheet": sheet.sheet_name,
                    "column": col_letter,
                    "header": header,
                }
            ))
    return chunks