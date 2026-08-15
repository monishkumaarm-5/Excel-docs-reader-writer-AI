#excel.py
from dataclasses import dataclass, field
from typing import List, Dict, Any

# --- Excel ---

@dataclass
class ExcelCell:
    row: str
    column: str
    data_excel: str

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