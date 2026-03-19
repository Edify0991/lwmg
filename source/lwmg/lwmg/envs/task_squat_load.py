from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SquatLoadTask:
    name: str = "g1_squat_load"
    description: str = "Squat-to-stand under bilateral hand payload."
