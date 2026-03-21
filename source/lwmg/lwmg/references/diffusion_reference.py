from __future__ import annotations

import torch

from .base_generator import BaseReferenceGenerator
from .reference_types import ReferenceTarget


class DiffusionReferenceGenerator(BaseReferenceGenerator):
    def generate(self, batch_size: int) -> ReferenceTarget:
        q = torch.randn(batch_size, 12) * 0.05
        return ReferenceTarget(joint_pos=q.cumsum(dim=0), joint_vel=torch.zeros_like(q))
