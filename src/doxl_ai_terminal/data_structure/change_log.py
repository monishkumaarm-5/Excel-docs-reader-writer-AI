# change_log.py
"""Change tracking with timestamps and rollback support."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class ChangeType(Enum):
    UPDATE = "update"
    ADD = "add"
    DELETE = "delete"


@dataclass
class Change:
    change_type: ChangeType
    location: Dict[str, Any]
    old_value: Optional[str]
    new_value: Optional[str]
    timestamp: str = field(default_factory=lambda: datetime.now().strftime("%H:%M:%S"))


@dataclass
class ChangeLog:
    changes: List[Change] = field(default_factory=list)

    def add(self, change_type: ChangeType, location: dict,
            old_value=None, new_value=None):
        self.changes.append(Change(
            change_type=change_type,
            location=location,
            old_value=str(old_value) if old_value is not None else None,
            new_value=str(new_value) if new_value is not None else None,
        ))

    def clear(self):
        self.changes.clear()

    def has_changes(self) -> bool:
        return len(self.changes) > 0

    def summary(self) -> str:
        if not self.changes:
            return "No changes recorded."
        lines = []
        for i, c in enumerate(self.changes, 1):
            loc = _format_location(c.location)
            if c.change_type == ChangeType.ADD:
                lines.append(f"  {i}. [{c.timestamp}] ADD    {loc} = '{c.new_value}'")
            elif c.change_type == ChangeType.UPDATE:
                lines.append(f"  {i}. [{c.timestamp}] UPDATE {loc}: '{c.old_value}' → '{c.new_value}'")
            elif c.change_type == ChangeType.DELETE:
                lines.append(f"  {i}. [{c.timestamp}] DELETE {loc} (was '{c.old_value}')")
        return "\n".join(lines)

    @property
    def count(self) -> int:
        return len(self.changes)

    def last_n(self, n: int = 5) -> List[Change]:
        """Return the last N changes."""
        return self.changes[-n:]


def _format_location(loc: Dict[str, Any]) -> str:
    """Human-readable location string from a location dict."""
    parts = []
    if "sheet" in loc:
        parts.append(loc["sheet"])
    if "column" in loc and "row" in loc:
        parts.append(f"[{loc['column']}{loc['row']}]")
    elif "row" in loc:
        parts.append(f"row {loc['row']}")
    elif "column" in loc:
        parts.append(f"col {loc['column']}")
    if "paragraph" in loc:
        parts.append(f"P{loc['paragraph']}")
    if "line" in loc:
        parts.append(f"L{loc['line']}")
    return "".join(parts) if parts else str(loc)
