from __future__ import annotations

import torch
from torch import nn


class ModeGating(nn.Module):
    """Softmax gating network for latent interaction/contact dynamic modes."""

    def __init__(
        self,
        state_dim: int,
        ref_dim: int,
        slow_latent_dim: int,
        fast_latent_dim: int,
        num_experts: int = 4,
        hidden_dim: int = 128,
        temperature: float = 1.0,
    ) -> None:
        super().__init__()
        self.num_experts = int(num_experts)
        self.temperature = float(max(1.0e-6, temperature))
        in_dim = state_dim + ref_dim + slow_latent_dim + fast_latent_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.num_experts),
        )

    def forward(
        self,
        state_features: torch.Tensor,
        reference_features: torch.Tensor,
        z_slow: torch.Tensor,
        z_fast: torch.Tensor,
    ) -> torch.Tensor:
        logits = self.net(torch.cat([state_features, reference_features, z_slow, z_fast], dim=-1))
        return torch.softmax(logits / self.temperature, dim=-1)
