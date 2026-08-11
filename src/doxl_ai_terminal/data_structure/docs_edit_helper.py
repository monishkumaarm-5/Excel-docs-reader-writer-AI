# ---- Find a line ----
from typing import List
from langchain_core.tools import tool
from doxl_ai_terminal.data_structure.docs import DocData, DocLine

@tool
def find_doc_line(doc_data: DocData, paragraph: int, line: int) -> DocLine:
    for l in doc_data.lines:
        if l.paragraph == paragraph and l.line == line:
            return l
    return None


# ---- Update a line ----
@tool
def update_doc_line(doc_data: DocData, paragraph: int, line: int, new_text: str):
    target = find_doc_line(doc_data, paragraph, line)
    if target:
        target.line_str = new_text
        print(f"Updated: P{paragraph} L{line} → '{new_text}'")
    else:
        print(f"Not found: P{paragraph} L{line}")


# ---- Add a new line ----
@tool
def add_doc_line(doc_data: DocData, paragraph: int, line: int, text: str):
    new_line = DocLine(paragraph=paragraph, line=line, line_str=text)
    doc_data.lines.append(new_line)
    doc_data.total_lines += 1
    print(f"Added: P{paragraph} L{line} → '{text}'")


# ---- Delete a line ----
@tool
def delete_doc_line(doc_data: DocData, paragraph: int, line: int):
    target = find_doc_line(doc_data, paragraph, line)
    if target:
        doc_data.lines.remove(target)
        doc_data.total_lines -= 1
        print(f"Deleted: P{paragraph} L{line}")
    else:
        print(f"Not found: P{paragraph} L{line}")


# ---- Search lines ----
@tool
def search_doc(doc_data: DocData, keyword: str) -> List[DocLine]:
    results = []
    for line in doc_data.lines:
        if keyword.lower() in line.line_str.lower():
            results.append(line)
    return results


# ---- Replace text in all lines ----
@tool
def replace_doc_text(doc_data: DocData, old_text: str, new_text: str):
    count = 0
    for line in doc_data.lines:
        if old_text in line.line_str:
            line.line_str = line.line_str.replace(old_text, new_text)
            count += 1
    print(f"Replaced '{old_text}' → '{new_text}' in {count} lines")