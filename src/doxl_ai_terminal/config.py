# config.py
"""Shared configuration constants."""

import os
from pathlib import Path


def _find_project_root() -> Path:
    current = Path(__file__).resolve().parent
    for parent in [current, *current.parents]:
        if (parent / "pyproject.toml").exists():
            return parent
    return Path(__file__).resolve().parent.parent.parent


# Vector store path (used by vector_config.py and vector_db_operation.py)
VECTOR_STORE_DIR: str = str(_find_project_root() / "vector_store")
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)