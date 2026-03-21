from __future__ import annotations

import torch


def anchor_gradient(current: torch.Tensor, anchor: torch.Tensor, weight: float) -> torch.Tensor:
    return weight * (current - anchor)
