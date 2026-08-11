from typing import Dict, List

from doxl_ai_terminal.data_structure.docs import DocData
from langchain_core.tools import tool

@tool
def write_word(doc_data: DocData, output_path: str):
    from docx import Document

    doc = Document()

    # Group lines by paragraph
    para_map: Dict[int, List[str]] = {}
    for line in doc_data.lines:
        if line.paragraph not in para_map:
            para_map[line.paragraph] = []
        para_map[line.paragraph].append(line.line_str)

    # Write each paragraph
    for p_num in sorted(para_map.keys()):
        text = "\n".join(para_map[p_num])
        doc.add_paragraph(text)

    doc.save(output_path)
    print(f"Saved: {output_path}")