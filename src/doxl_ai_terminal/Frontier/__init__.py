"""
Frontier sub-package.

Live document managers for reading, editing, and syncing docs/excel files.
"""

from doxl_ai_terminal.Frontier.ChangeLogManager_docs import LiveDocManager
from doxl_ai_terminal.Frontier.changelogmanager_excel import LiveExcelManager
from doxl_ai_terminal.Frontier.manager_tools import load_file

__all__ = [
    "LiveDocManager",
    "LiveExcelManager",
    "load_file",
]
