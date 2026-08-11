"""Document processing pipeline orchestrator."""
import os
from typing import List

from src.doxl_ai_terminal.Chunker.chunking import chunk_doc_sub_para, chunk_doc_line_by_line, chunk_excel_row_wise, \
    chunk_excel_column_wise
from src.doxl_ai_terminal.Frontier.fileReader import read_word, read_excel
from src.doxl_ai_terminal.data_handler.vector_config import store_in_chroma
from src.doxl_ai_terminal.data_structure.excel import VectorDBInstance


def process_file(filepath) -> List[VectorDBInstance]:
    ext = os.path.splitext(filepath)[1].lower()
    filename = os.path.splitext(os.path.basename(filepath))[0]
    db_instances = []

    if ext == ".docx":
        data = read_word(filepath)
        label = "doc"
        format_map = {
            "line_by_line": chunk_doc_line_by_line,
            "sub_para":     chunk_doc_sub_para,
        }

    elif ext == ".xlsx":
        data = read_excel(filepath)
        label = "excel"
        format_map = {
            "row_wise":    chunk_excel_row_wise,
            "column_wise": chunk_excel_column_wise,
        }

    else:
        print(f"Unsupported: {ext}")
        return []

    for format_name, chunk_fn in format_map.items():
        print(f"  Chunking: {format_name}...")

        chunks = chunk_fn(data)

        if not chunks:
            continue

        collection_name = f"{filename}_{format_name}"
        vectorstore, retriever = store_in_chroma(chunks, collection_name)

        db_instances.append(VectorDBInstance(
            label=label,
            format_name=format_name,
            collection_name=collection_name,
            total_chunks=len(chunks),
            vectorstore=vectorstore,
            retriever=retriever,
        ))

        print(f"  Stored {len(chunks)} chunks → '{collection_name}'")

    return db_instances