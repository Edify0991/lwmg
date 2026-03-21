from __future__ import annotations

import torch
from torch import nn


class InteractionEncoder(nn.Module):
    def __init__(self, state_dim: int, ref_dim: int, latent_dim: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(state_dim + ref_dim + state_dim, 128), nn.ReLU(), nn.Linear(128, latent_dim))

    def forward(self, history: torch.Tensor, r_t: torch.Tensor, s_nom_tp1: torch.Tensor) -> torch.Tensor:
        h = history.mean(dim=1)
        return self.net(torch.cat([h, r_t, s_nom_tp1], dim=-1))
