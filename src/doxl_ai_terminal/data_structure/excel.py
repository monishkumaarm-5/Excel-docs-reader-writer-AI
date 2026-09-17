# excel.py
"""Data structures for Excel spreadsheet representation.

Key design decision: ExcelCell.row is stored as an int (not a string)
to eliminate the constant int(cell.row) / str(cell.row) conversions
that plagued the previous version. All internal code works with int
rows; string conversion happens only at the boundary (user-facing
display, LLM tool arguments).
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Union


def coerce_excel_value(value: Any) -> Any:
    """Best-effort: turn a numeric-looking string into a real int/float
    so a number round-trips to Excel as a number instead of text.

    Values that are already typed (int, float, bool, datetime, ...) are
    returned untouched. Strings with a protective leading zero (e.g.
    "00501" or "007") are deliberately left as text.
    """
    if not isinstance(value, str):
        return value

    s = value.strip()
    if s == "" or s.startswith("="):
        return value

    body = s[1:] if s and s[0] in "+-" else s
    if len(body) > 1 and body[0] == "0" and body[1] != ".":
        return value

    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        return value


def parse_row(value: Any) -> int:
    """Safely parse a row value to int. Accepts int, str, or float."""
    if isinstance(value, int):
        return value
    return int(str(value).strip())


@dataclass
class ExcelCell:
    row: int                 # Row number (1-based integer)
    column: str              # Column letter (e.g. "A", "B", "AA")
    data_excel: Union[str, int, float, bool, Any]
    # --- Formatting (all optional / tri-state: None = "leave alone") ---
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    font_color: Optional[str] = None
    fill_color: Optional[str] = None
    alignment: Optional[str] = None
    number_format: Optional[str] = None

    @property
    def row_str(self) -> str:
        """Row as a string for display/serialization."""
        return str(self.row)

    @property
    def address(self) -> str:
        """Cell address like 'B3'."""
        return f"{self.column}{self.row}"


@dataclass
class ExcelSheet:
    sheet_name: str
    total_rows: int
    total_columns: int
    cells: List[ExcelCell] = field(default_factory=list)

    def get_cell(self, row: int, column: str) -> Optional[ExcelCell]:
        """Find a cell by position, or None."""
        for cell in self.cells:
            if cell.row == row and cell.column == column:
                return cell
        return None

    def get_row_cells(self, row: int) -> List[ExcelCell]:
        """All cells in a given row, sorted by column."""
        return sorted(
            [c for c in self.cells if c.row == row],
            key=lambda c: c.column,
        )

    def get_column_cells(self, column: str) -> List[ExcelCell]:
        """All cells in a given column, sorted by row."""
        return sorted(
            [c for c in self.cells if c.column == column],
            key=lambda c: c.row,
        )


@dataclass
class ExcelData:
    filename: str
    total_sheets: int
    sheets: List[ExcelSheet] = field(default_factory=list)

    def get_sheet(self, name: str) -> Optional[ExcelSheet]:
        """Find a sheet by name, or None."""
        for sheet in self.sheets:
            if sheet.sheet_name == name:
                return sheet
        return None


# --- Chunk ---

@dataclass
class Chunk:
    chunk_id: str
    text: str
    metadata: Dict[str, Any]


# --- Vector DB Instance ---

@dataclass
class VectorDBInstance:
    label: str
    format_name: str
    collection_name: str
    total_chunks: int
    vectorstore: Any
    retriever: Any
