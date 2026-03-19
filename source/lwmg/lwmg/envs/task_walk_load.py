from __future__ import annotations

from dataclasses import dataclass


@dataclass
class WalkLoadTask:
    name: str = "g1_walk_load"
    description: str = "Forward walking under bilateral hand payload."
