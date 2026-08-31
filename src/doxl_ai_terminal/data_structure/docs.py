from dataclasses import dataclass, field
from typing import List, Optional

# --- Word Document Structure ---

@dataclass
class DocLine:
    paragraph: int
    line: int
    line_str: str
    # --- Formatting (all optional / tri-state: None = "not specified,
    # leave whatever is already there alone") ---
    font_name: Optional[str] = None
    font_size: Optional[float] = None    # points
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    font_color: Optional[str] = None     # 6-digit hex RGB, e.g. "FF0000"
    alignment: Optional[str] = None      # "left" | "center" | "right" | "justify"

@dataclass
class DocData:
    filename: str
    total_paragraphs: int
    total_lines: int
    lines: List[DocLine] = field(default_factory=list)