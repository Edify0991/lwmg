from __future__ import annotations

import torch
from torch import nn


class _ResidualExpert(nn.Module):
    def __init__(self, in_dim: int, state_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, state_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ResidualExperts(nn.Module):
    """Mixture-of-experts residual field for interaction-induced state deltas."""

    def __init__(
        self,
        state_dim: int,
        ref_dim: int,
        ctrl_dim: int,
        slow_latent_dim: int,
        fast_latent_dim: int,
        num_experts: int = 4,
        hidden_dim: int = 128,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.slow_latent_dim = int(slow_latent_dim)
        self.fast_latent_dim = int(fast_latent_dim)
        self.num_experts = int(num_experts)

        in_dim = state_dim + ref_dim + ctrl_dim + slow_latent_dim + fast_latent_dim
        self.experts = nn.ModuleList([_ResidualExpert(in_dim, state_dim, hidden_dim) for _ in range(self.num_experts)])

    def _split_if_needed(self, z_slow: torch.Tensor, z_fast: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        if z_fast is not None:
            return z_slow, z_fast
        total = self.slow_latent_dim + self.fast_latent_dim
        if z_slow.shape[-1] != total:
            raise ValueError(
                f"Expected concatenated latent dim {total} when z_fast is None, got {z_slow.shape[-1]}."
            )
        return z_slow[..., : self.slow_latent_dim], z_slow[..., self.slow_latent_dim :]

    def forward(
        self,
        s_t: torch.Tensor,
        r_t: torch.Tensor,
        u_t: torch.Tensor,
        z_slow: torch.Tensor,
        z_fast: torch.Tensor | None = None,
        mode_weights: torch.Tensor | None = None,
        return_per_expert: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        z_slow_use, z_fast_use = self._split_if_needed(z_slow, z_fast)
        x = torch.cat([s_t, r_t, u_t, z_slow_use, z_fast_use], dim=-1)
        per_expert = torch.stack([expert(x) for expert in self.experts], dim=1)
        if mode_weights is None:
            mode_weights = torch.full(
                (x.shape[0], self.num_experts),
                1.0 / float(self.num_experts),
                dtype=x.dtype,
                device=x.device,
            )
        delta = (mode_weights.unsqueeze(-1) * per_expert).sum(dim=1)
        if return_per_expert:
            return delta, per_expert
        return delta
