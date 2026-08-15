"""
Pipeline sub-package.

Contains processing pipelines for document operations.
"""

from doxl_ai_terminal.pipeline.pipeliner import process_file
from doxl_ai_terminal.pipeline.search import search

__all__ = [
    "process_file",
    "search",
]
