from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class PrivilegedLoadLabel:
    payload_mass: float
    payload_com_shift: torch.Tensor
    hand_wrench_lr: torch.Tensor
    failure_type: str


class RandomLoadSampler:
    def __init__(self, mass_range: tuple[float, float] = (0.0, 12.0)) -> None:
        self.mass_range = mass_range

    def sample(self) -> PrivilegedLoadLabel:
        payload_mass = float(torch.empty(1).uniform_(*self.mass_range).item())
        payload_com_shift = torch.empty(3).uniform_(-0.08, 0.08)
        hand_wrench_lr = torch.empty(2, 3).uniform_(-30.0, 30.0)
        return PrivilegedLoadLabel(payload_mass, payload_com_shift, hand_wrench_lr, "none")
