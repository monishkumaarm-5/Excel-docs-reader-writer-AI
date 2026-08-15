# ---- Find a line ----
from typing import List, Optional
from doxl_ai_terminal.data_structure.docs import DocData, DocLine


def find_doc_line(doc_data: DocData, paragraph: int, line: int) -> Optional[DocLine]:
    """Internal helper — find a line by paragraph and line number."""
    for l in doc_data.lines:
        if l.paragraph == paragraph and l.line == line:
            return l
    return None


# ---- Update a line ----
def update_doc_line(doc_data: DocData, paragraph: int, line: int, new_text: str) -> str:
    """Update a specific line in a Word document by paragraph and line number."""
    target = find_doc_line(doc_data, paragraph, line)
    if target:
        old = target.line_str
        target.line_str = new_text
        return f"Updated: P{paragraph} L{line} '{old}' → '{new_text}'"
    return f"Not found: P{paragraph} L{line}"


# ---- Add a new line ----
def add_doc_line(doc_data: DocData, paragraph: int, line: int, text: str) -> str:
    """Add a new line to a Word document at a specific paragraph and line position."""
    new_line = DocLine(paragraph=paragraph, line=line, line_str=text)
    doc_data.lines.append(new_line)
    doc_data.total_lines += 1
    return f"Added: P{paragraph} L{line} → '{text}'"


# ---- Delete a line ----
def delete_doc_line(doc_data: DocData, paragraph: int, line: int) -> str:
    """Delete a line from a Word document by paragraph and line number."""
    target = find_doc_line(doc_data, paragraph, line)
    if target:
        doc_data.lines.remove(target)
        doc_data.total_lines -= 1
        return f"Deleted: P{paragraph} L{line}"
    return f"Not found: P{paragraph} L{line}"


# ---- Search lines ----
def search_doc(doc_data: DocData, keyword: str) -> List[DocLine]:
    """Search for a keyword across all lines in the Word document."""
    results = []
    for line in doc_data.lines:
        if keyword.lower() in line.line_str.lower():
            results.append(line)
    return results


# ---- Replace text in all lines ----
def replace_doc_text(doc_data: DocData, old_text: str, new_text: str) -> str:
    """Find and replace text across all lines in the Word document."""
    count = 0
    for line in doc_data.lines:
        if old_text in line.line_str:
            line.line_str = line.line_str.replace(old_text, new_text)
            count += 1
    return f"Replaced '{old_text}' → '{new_text}' in {count} lines"