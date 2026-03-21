from __future__ import annotations

import torch


def softplus_barrier(x: torch.Tensor, margin: float = 0.0, beta: float = 10.0) -> torch.Tensor:
    return torch.nn.functional.softplus(beta * (x - margin)) / beta
