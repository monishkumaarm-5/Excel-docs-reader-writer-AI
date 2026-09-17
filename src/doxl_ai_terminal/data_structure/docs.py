# docs.py
"""Data structures for Word document representation."""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class DocLine:
    paragraph: int
    line: int
    line_str: str
    # --- Formatting (all optional / tri-state: None = "leave alone") ---
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    strikethrough: Optional[bool] = None
    font_color: Optional[str] = None
    highlight_color: Optional[str] = None
    superscript: Optional[bool] = None
    subscript: Optional[bool] = None
    alignment: Optional[str] = None
    style: Optional[str] = None
    heading_level: Optional[int] = None  # Derived from style, read-only


@dataclass
class DocData:
    filename: str
    total_paragraphs: int
    total_lines: int
    lines: List[DocLine] = field(default_factory=list)

    def get_paragraph_lines(self, paragraph: int) -> List[DocLine]:
        """All lines in a given paragraph, sorted by line number."""
        return sorted(
            [l for l in self.lines if l.paragraph == paragraph],
            key=lambda l: l.line,
        )

    def get_line(self, paragraph: int, line: int) -> Optional[DocLine]:
        """Find a specific line, or None."""
        for l in self.lines:
            if l.paragraph == paragraph and l.line == line:
                return l
        return None
