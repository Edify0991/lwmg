from __future__ import annotations

import torch
from torch import nn


class NominalTransitionModel(nn.Module):
    def __init__(self, state_dim: int, ref_dim: int, ctrl_dim: int, hidden_dim: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + ref_dim + ctrl_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, s_t: torch.Tensor, r_t: torch.Tensor, u_t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_t, r_t, u_t], dim=-1))
