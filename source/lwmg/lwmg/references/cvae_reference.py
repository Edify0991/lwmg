from __future__ import annotations

import torch

from .base_generator import BaseReferenceGenerator
from .reference_types import ReferenceTarget


class CVAEReferenceGenerator(BaseReferenceGenerator):
    def generate(self, batch_size: int) -> ReferenceTarget:
        q = torch.randn(batch_size, 12) * 0.1
        return ReferenceTarget(joint_pos=q, joint_vel=torch.zeros_like(q))
