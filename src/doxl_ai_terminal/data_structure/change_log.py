from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum


class ChangeType(Enum):
    UPDATE = "update"
    ADD = "add"
    DELETE = "delete"


@dataclass
class Change:
    change_type: ChangeType
    location: Dict[str, Any]   # where the change happened
    old_value: Optional[str]
    new_value: Optional[str]


@dataclass
class ChangeLog:
    changes: List[Change] = field(default_factory=list)

    def add(self, change_type: ChangeType, location: dict, old_value=None, new_value=None):
        self.changes.append(Change(
            change_type=change_type,
            location=location,
            old_value=old_value,
            new_value=new_value,
        ))

    def clear(self):
        self.changes.clear()

    def has_changes(self) -> bool:
        return len(self.changes) > 0

    def summary(self) -> str:
        lines = []
        for c in self.changes:
            lines.append(f"  [{c.change_type.value}] {c.location} : '{c.old_value}' → '{c.new_value}'")
        return "\n".join(lines)