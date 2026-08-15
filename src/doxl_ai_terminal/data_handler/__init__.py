"""
Data handler sub-package.

Contains vector database configuration and operations.
"""

from doxl_ai_terminal.data_handler.vector_config import get_embedding_model, store_in_chroma
from doxl_ai_terminal.data_handler.vector_db_operation import VectorDBManager

__all__ = [
    "get_embedding_model",
    "store_in_chroma",
    "VectorDBManager",
]
