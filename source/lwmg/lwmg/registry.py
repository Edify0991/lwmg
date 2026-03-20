from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict


@dataclass
class Registry:
    """Simple name->object registry used by experiments."""

    entries: Dict[str, Any] = field(default_factory=dict)

    def register(self, name: str, obj: Any) -> None:
        self.entries[name] = obj

    def build(self, name: str) -> Any:
        return self.entries[name]
