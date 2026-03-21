from __future__ import annotations

import torch
from torch import nn


class FlowDecoder(nn.Module):
    def __init__(self, latent_dim: int, ref_dim: int) -> None:
        super().__init__()
        self.net = nn.Linear(latent_dim, ref_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
