from __future__ import annotations

import torch


def jacobian_transpose_torque(jacobian: torch.Tensor, wrench: torch.Tensor) -> torch.Tensor:
    return jacobian.transpose(-1, -2) @ wrench.unsqueeze(-1)
