from __future__ import annotations

import torch
import torch.nn.functional as F


def task_cost(progress: torch.Tensor, target_vel_err: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(progress, torch.ones_like(progress)) + F.huber_loss(target_vel_err, torch.zeros_like(target_vel_err))


def tracking_feasibility_cost(pred_tracking_error: torch.Tensor) -> torch.Tensor:
    return F.softplus(pred_tracking_error).mean()


def stability_cost(trunk_tilt: torch.Tensor, base_height: torch.Tensor, support_margin: torch.Tensor, slip: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
    tilt = F.softplus(torch.abs(trunk_tilt) - 0.6).mean()
    height = F.softplus(0.35 - base_height).mean()
    support = F.softplus(0.1 - support_margin).mean()
    slip_term = F.softplus(slip).mean()
    torque_term = F.softplus(torch.abs(torque) - 1.0).mean()
    return tilt + height + support + slip_term + torque_term


def smoothness_cost(reference: torch.Tensor) -> torch.Tensor:
    d = reference[:, 1:] - reference[:, :-1]
    return (d * d).mean()


def uncertainty_cost(variance: torch.Tensor) -> torch.Tensor:
    return variance.mean()


def prior_anchor_cost(reference: torch.Tensor, unguided_reference: torch.Tensor) -> torch.Tensor:
    return F.huber_loss(reference, unguided_reference)
