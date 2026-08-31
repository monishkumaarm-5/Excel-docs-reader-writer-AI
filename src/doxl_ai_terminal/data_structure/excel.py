#excel.py
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional

# --- Excel ---

@dataclass
class ExcelCell:
    row: str
    column: str
    data_excel: str
    # --- Formatting (all optional / tri-state: None = "not specified,
    # leave whatever is already there alone") ---
    font_name: Optional[str] = None
    font_size: Optional[float] = None
    bold: Optional[bool] = None
    italic: Optional[bool] = None
    underline: Optional[bool] = None
    font_color: Optional[str] = None     # 6-digit hex RGB, e.g. "FF0000"
    fill_color: Optional[str] = None     # 6-digit hex RGB background fill
    alignment: Optional[str] = None      # "left" | "center" | "right" | "justify"
    number_format: Optional[str] = None  # e.g. "0.00", "yyyy-mm-dd", "$#,##0.00"

@dataclass
class ExcelSheet:
    sheet_name: str
    total_rows: int
    total_columns: int
    cells: List[ExcelCell] = field(default_factory=list)

@dataclass
class ExcelData:
    filename: str
    total_sheets: int
    sheets: List[ExcelSheet] = field(default_factory=list)

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
    vectorstore: Any    # LangChain Chroma instance
    retriever: Any      # LangChain retriever