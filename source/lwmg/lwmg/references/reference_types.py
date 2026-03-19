from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ReferenceTarget:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
