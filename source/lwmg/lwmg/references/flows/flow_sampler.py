from __future__ import annotations

import torch


def init_latent(batch_size: int, horizon: int, latent_dim: int, device: torch.device) -> torch.Tensor:
    return torch.randn(batch_size, horizon, latent_dim, device=device)
