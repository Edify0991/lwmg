from __future__ import annotations

import torch
import torch.nn.functional as F


def nominal_state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def nominal_rollout_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def residual_state_loss(pred_residual: torch.Tensor, target_residual: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_residual, target_residual)


def residual_rollout_loss(pred_rollout: torch.Tensor, target_rollout: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_rollout, target_rollout)


def residual_zero_loss(pred_residual: torch.Tensor) -> torch.Tensor:
    return (pred_residual * pred_residual).mean()


def paired_counterfactual_loss(pred_loaded: torch.Tensor, pred_nominal: torch.Tensor, true_loaded: torch.Tensor, true_nominal: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_loaded - pred_nominal, true_loaded - true_nominal)


def auxiliary_failure_contact_loss(logit_failure: torch.Tensor, tgt_failure: torch.Tensor, logit_contact: torch.Tensor, tgt_contact: torch.Tensor, enabled: bool = False) -> torch.Tensor:
    if not enabled:
        return torch.zeros((), device=logit_failure.device)
    return F.binary_cross_entropy_with_logits(logit_failure, tgt_failure) + F.binary_cross_entropy_with_logits(logit_contact, tgt_contact)
