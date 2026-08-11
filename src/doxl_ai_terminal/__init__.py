"""
doxl-ai-terminal

AI-powered terminal for reading and writing Excel & Word documents.

Usage::

    >>> import doxl_ai_terminal
    >>> print(doxl_ai_terminal.__version__)
    '0.1.0'

    >>> from doxl_ai_terminal.agents import excel_agent, docs_agent
"""

__version__ = "0.1.0"
__author__ = "Mohan"

from .agents import config_agent, docs_agent, excel_agent, terminal_agent
from .data_handler import vector_config, vector_db_operation
from .pipeline import pipeliner

__all__ = [
    "__version__",
    "config_agent",
    "docs_agent",
    "excel_agent",
    "terminal_agent",
    "vector_config",
    "vector_db_operation",
    "pipeliner",
]
