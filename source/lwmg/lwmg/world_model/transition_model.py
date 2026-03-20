from __future__ import annotations

import torch
from torch import nn


class TransitionModel(nn.Module):
    def __init__(self, state_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, state: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([state, z], dim=-1))
