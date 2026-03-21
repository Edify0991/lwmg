from __future__ import annotations

import torch
from torch import nn


class ResidualTransitionModel(nn.Module):
    def __init__(self, state_dim: int, ref_dim: int, ctrl_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + ref_dim + ctrl_dim + latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, state_dim),
        )

    def forward(self, s_t: torch.Tensor, r_t: torch.Tensor, u_t: torch.Tensor, z_int: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([s_t, r_t, u_t, z_int], dim=-1))
