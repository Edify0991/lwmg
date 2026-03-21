from __future__ import annotations

import torch

from .base_generator import BaseReferenceGenerator
from .reference_types import ReferenceTarget


class FlowReferenceGenerator(BaseReferenceGenerator):
    def generate(self, batch_size: int) -> ReferenceTarget:
        base = torch.linspace(0, 1, steps=batch_size).unsqueeze(-1)
        q = base.repeat(1, 12)
        return ReferenceTarget(joint_pos=q, joint_vel=torch.gradient(q, dim=0)[0])
