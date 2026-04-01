from __future__ import annotations

from dataclasses import dataclass, field

import torch


@dataclass
class ReferenceTarget:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    root_pos: torch.Tensor | None = None
    root_quat: torch.Tensor | None = None
    frame_idx: torch.Tensor | None = None
    extras: dict[str, torch.Tensor] = field(default_factory=dict)
