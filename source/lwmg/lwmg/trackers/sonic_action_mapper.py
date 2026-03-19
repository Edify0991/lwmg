from __future__ import annotations

import torch


def map_sonic_output_to_targets(raw_action: torch.Tensor, target_dim: int) -> torch.Tensor:
    if raw_action.numel() < target_dim:
        pad = torch.zeros(target_dim - raw_action.numel(), dtype=raw_action.dtype)
        return torch.cat([raw_action, pad], dim=0)
    return raw_action[:target_dim]
