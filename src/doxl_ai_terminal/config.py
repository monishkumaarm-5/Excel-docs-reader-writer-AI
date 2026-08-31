# config.py
"""Shared configuration constants."""

import os
from pathlib import Path
from typing import Optional


def _find_project_root() -> Optional[Path]:
    """Look for a pyproject.toml above this file (source / editable install)."""
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return None


def _default_data_dir() -> Path:
    """Where to store the vector DB and cached embedding model.

    - Running from a source checkout (or ``pip install -e .``): use the
      project root, so everything stays inside the repo — this matches
      the ``vector_store/`` and ``models/`` entries in .gitignore.
    - Running from a real ``pip install``/PyPI install: there is no
      pyproject.toml anywhere near the installed package, so falling
      back to a path inside site-packages (as the old code did) would
      be unwritable on many systems and the wrong place for a growing
      SQLite vector store + embedding-model cache anyway. Use a
      per-user data directory instead, consistent with
      credential_store.py's existing ``~/.docs-excel/`` convention.

    Either path can be overridden with the DOXL_AI_DATA_DIR env var.
    """
    override = os.environ.get("DOXL_AI_DATA_DIR")
    if override:
        return Path(override).expanduser()

    project_root = _find_project_root()
    if project_root is not None:
        return project_root

    return Path.home() / ".docs-excel"


_DATA_DIR = _default_data_dir()

# Vector store path (used by vector_config.py and vector_db_operation.py)
VECTOR_STORE_DIR: str = str(_DATA_DIR / "vector_store")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)