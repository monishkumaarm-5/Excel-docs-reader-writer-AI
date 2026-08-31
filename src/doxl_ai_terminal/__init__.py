"""
doxl-ai-terminal

AI-powered terminal for reading and writing Excel & Word documents.

Usage::

    >>> import doxl_ai_terminal
    >>> print(doxl_ai_terminal.__version__)
    '0.1.0'

    >>> from doxl_ai_terminal.data_structure import DocData, ExcelData, Chunk
    >>> from doxl_ai_terminal.tools import ALL_TOOLS, load_file
    >>> from doxl_ai_terminal.Frontier import LiveDocManager, LiveExcelManager
"""

__version__ = "0.1.0"
__author__ = "Monish Kumaar M"

# Lazy imports — heavy dependencies (LangChain, Chroma, HuggingFace) are
# deferred until actually used.  This keeps `import doxl_ai_terminal` fast.

__all__ = [
    "__version__",
    "config",
    "data_structure",
    "data_handler",
    "pipeline",
    "Frontier",
    "Chunker",
    "tools",
]
