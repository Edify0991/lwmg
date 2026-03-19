from __future__ import annotations

from dataclasses import dataclass


@dataclass
class SonicMotionFormat:
    frequency_hz: int = 50
    joint_order: str = "isaaclab_g1"
