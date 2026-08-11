from dataclasses import dataclass, field
from typing import List

# --- Word Document Structure ---

@dataclass
class DocLine:
    paragraph: int
    line: int
    line_str: str

@dataclass
class DocData:
    filename: str
    total_paragraphs: int
    total_lines: int
    lines: List[DocLine] = field(default_factory=list)