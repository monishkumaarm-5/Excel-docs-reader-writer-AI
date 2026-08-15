#chunking
# ---- Doc: Line by Line ----
from typing import Dict, List

from doxl_ai_terminal.data_structure.docs import DocData
from doxl_ai_terminal.data_structure.excel import Chunk, ExcelData

# Rough safety margin under the embedding model's 256 wordpiece-token
# limit (all-MiniLM-L6-v2). Wordpiece tokenization typically expands
# whitespace-word count by ~1.3x, so capping at ~150 words keeps a
# chunk comfortably inside the window instead of having its tail
# silently dropped by the embedder.
MAX_CHUNK_WORDS = 150

# A single Excel column can hold lakhs of rows. Cramming all of them
# into one chunk would (a) get silently truncated at the embedding
# model's context window — only the first ~150 words would ever be
# "seen" — and (b) be slow to embed for no benefit. Instead each
# column is windowed into chunks of this many rows, so every value in
# a huge column stays reachable by vector search.
COLUMN_CHUNK_WINDOW = 40


def _split_into_windows(items: List, window_size: int) -> List[List]:
    """Split a list into consecutive windows of at most window_size items."""
    if not items:
        return []
    return [items[i:i + window_size] for i in range(0, len(items), window_size)]


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
        # Very long paragraphs get split into multiple sub-chunks so
        # each one stays inside the embedding model's context window
        # instead of being silently truncated.
        word_windows = _split_into_windows(" ".join(lines).split(), MAX_CHUNK_WORDS)
        total_parts = len(word_windows)

        for part_idx, words in enumerate(word_windows, start=1):
            chunk_id = f"{doc_data.filename}_p{p_num}"
            if total_parts > 1:
                chunk_id += f"_part{part_idx}"
            chunks.append(Chunk(
                chunk_id=chunk_id,
                text=" ".join(words),
                metadata={
                    "source": doc_data.filename,
                    "type": "doc",
                    "format": "sub_para",
                    "paragraph": p_num,
                    "total_lines": len(lines),
                    "part": part_idx,
                    "total_parts": total_parts,
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
        # Keep (row, value) pairs — not just bare values — so each
        # windowed chunk can record which rows it actually covers.
        col_data: Dict[str, List[tuple]] = {}

        for cell in sheet.cells:
            if cell.row == "1":
                headers[cell.column] = cell.data_excel
            else:
                if cell.column not in col_data:
                    col_data[cell.column] = []
                col_data[cell.column].append((cell.row, cell.data_excel))

        for col_letter, row_values in col_data.items():
            header = headers.get(col_letter, col_letter)
            windows = _split_into_windows(row_values, COLUMN_CHUNK_WINDOW)
            total_parts = len(windows)

            for part_idx, window in enumerate(windows, start=1):
                row_start, row_end = window[0][0], window[-1][0]
                values_text = ", ".join(f"row{r}={v}" for r, v in window)
                chunk_id = (
                    f"{excel_data.filename}_{sheet.sheet_name}"
                    f"_col{col_letter}_part{part_idx}"
                )
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    text=f"{header} (rows {row_start}-{row_end}): {values_text}",
                    metadata={
                        "source": excel_data.filename,
                        "type": "excel",
                        "format": "column_wise",
                        "sheet": sheet.sheet_name,
                        "column": col_letter,
                        "header": header,
                        "row_start": row_start,
                        "row_end": row_end,
                        "part": part_idx,
                        "total_parts": total_parts,
                    }
                ))
    return chunks