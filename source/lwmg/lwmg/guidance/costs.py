from __future__ import annotations

import torch


def guidance_cost(
    task_progress: torch.Tensor,
    tracking_feasibility: torch.Tensor,
    hard_risk: torch.Tensor,
    soft_risk: torch.Tensor,
    smoothness: torch.Tensor,
    torque_penalty: torch.Tensor,
) -> torch.Tensor:
    return (
        -task_progress
        + 0.5 * (1.0 - tracking_feasibility)
        + 3.0 * hard_risk
        + 1.5 * soft_risk
        + 0.1 * smoothness
        + 0.2 * torque_penalty
    )
