from __future__ import annotations

import torch


def ensemble_variance(predictions: torch.Tensor) -> torch.Tensor:
    # [E, B, T, D] -> [B, T, D]
    return torch.var(predictions, dim=0, unbiased=False)
