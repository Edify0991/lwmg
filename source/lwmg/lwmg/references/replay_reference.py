from __future__ import annotations

import torch

from .base_generator import BaseReferenceGenerator
from .reference_types import ReferenceTarget


class ReplayReferenceGenerator(BaseReferenceGenerator):
    def generate(self, batch_size: int) -> ReferenceTarget:
        return ReferenceTarget(joint_pos=torch.zeros(batch_size, 12), joint_vel=torch.zeros(batch_size, 12))
