from __future__ import annotations

import torch
import torch.nn.functional as F


def flow_objective(pred_velocity: torch.Tensor, target_velocity: torch.Tensor, family: str = "flow_matching") -> torch.Tensor:
    """Generic L_flow(theta; psi), psi encoded as `family`."""
    if family in {"flow_matching", "rectified_flow", "mean_flow"}:
        return F.mse_loss(pred_velocity, target_velocity)
    raise ValueError(f"Unsupported flow family: {family}")
