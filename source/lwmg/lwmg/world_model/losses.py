from __future__ import annotations

import torch
import torch.nn.functional as F


def state_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def rollout_loss(pred_seq: torch.Tensor, target_seq: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_seq, target_seq)


def hard_failure_bce(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target)


def soft_failure_bce(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target)


def contact_bce(logit: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.binary_cross_entropy_with_logits(logit, target)


def weak_context_loss(pred_mass: torch.Tensor, pred_com: torch.Tensor, target_mass: torch.Tensor, target_com: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_mass, target_mass) + F.mse_loss(pred_com, target_com)


def hand_wrench_prediction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.l1_loss(pred, target)


def arm_consistency_loss(jacobian: torch.Tensor, wrench: torch.Tensor, torque: torch.Tensor) -> torch.Tensor:
    projected = (jacobian.transpose(-1, -2) @ wrench.unsqueeze(-1)).squeeze(-1)
    return F.mse_loss(projected, torque)


def foot_stability_loss(slip: torch.Tensor, support_margin: torch.Tensor, torque: torch.Tensor, torque_limit: float = 1.0) -> torch.Tensor:
    torque_penalty = torch.relu(torch.abs(torque) - torque_limit).mean()
    return slip.mean() + (1.0 - support_margin).mean() + torque_penalty
