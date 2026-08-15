#credential.store.py
"""
Credential storage — persists API key and model selection to disk.

Stores credentials in ~/.docs-excel/config.json so the user only
needs to enter them once. Subsequent runs load from disk automatically.

Usage:
    from doxl_ai_terminal.pipeline.credential_store import (
        has_credentials, load_credentials, save_credentials, clear_credentials,
    )
"""

import json
import os
from pathlib import Path


# Config directory: ~/.docs-excel/
CONFIG_DIR = Path.home() / ".docs-excel"
CONFIG_FILE = CONFIG_DIR / "config.json"


def _ensure_config_dir():
    """Create the config directory if it doesn't exist."""
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)


def has_credentials() -> bool:
    """Check if stored credentials exist on disk."""
    if not CONFIG_FILE.exists():
        return False
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        return bool(data.get("api_key")) and bool(data.get("model"))
    except (json.JSONDecodeError, OSError):
        return False


def load_credentials() -> dict | None:
    """Load stored credentials from disk.

    Returns:
        {"api_key": "...", "model": "..."} or None if not found.
    """
    if not CONFIG_FILE.exists():
        return None
    try:
        data = json.loads(CONFIG_FILE.read_text(encoding="utf-8"))
        if data.get("api_key") and data.get("model"):
            return data
        return None
    except (json.JSONDecodeError, OSError):
        return None


def save_credentials(api_key: str, model: str) -> None:
    """Save credentials to disk."""
    _ensure_config_dir()
    data = {"api_key": api_key, "model": model}
    CONFIG_FILE.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def clear_credentials() -> None:
    """Delete stored credentials."""
    if CONFIG_FILE.exists():
        CONFIG_FILE.unlink()
