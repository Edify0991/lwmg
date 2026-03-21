from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class PredictionOutputs:
    next_state: torch.Tensor
    hard_failure_logit: torch.Tensor
    soft_failure_logit: torch.Tensor
    contact_logit: torch.Tensor
    hand_wrench: torch.Tensor
    payload_mass: torch.Tensor
    payload_com_shift: torch.Tensor


class PredictionHeads(nn.Module):
    def __init__(self, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.hard = nn.Linear(state_dim, 1)
        self.soft = nn.Linear(state_dim, 1)
        self.contact = nn.Linear(state_dim, 2)
        self.wrench = nn.Linear(state_dim, 6)
        self.mass = nn.Linear(state_dim, 1)
        self.com = nn.Linear(state_dim, 3)

    def forward(self, h: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "hard_failure_logit": self.hard(h),
            "soft_failure_logit": self.soft(h),
            "contact_logit": self.contact(h),
            "hand_wrench": self.wrench(h),
            "payload_mass": self.mass(h),
            "payload_com_shift": self.com(h),
        }
