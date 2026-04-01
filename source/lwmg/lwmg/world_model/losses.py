from __future__ import annotations

import torch
import torch.nn.functional as F

from .energy_support_consistency_loss import EnergySupportConsistencyLoss


_DEFAULT_ENERGY_SUPPORT = EnergySupportConsistencyLoss()


def nominal_state_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def nominal_rollout_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred, target)


def residual_state_loss(pred_residual: torch.Tensor, target_residual: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_residual, target_residual)


def residual_rollout_loss(pred_rollout: torch.Tensor, target_rollout: torch.Tensor) -> torch.Tensor:
    return F.mse_loss(pred_rollout, target_rollout)


def residual_zero_loss(pred_residual: torch.Tensor) -> torch.Tensor:
    return pred_residual.pow(2).mean()


def paired_counterfactual_loss(
    pred_loaded: torch.Tensor,
    pred_nominal: torch.Tensor,
    true_loaded: torch.Tensor,
    true_nominal: torch.Tensor,
) -> torch.Tensor:
    return F.mse_loss(pred_loaded - pred_nominal, true_loaded - true_nominal)


def support_consistency_loss(
    support_margin: torch.Tensor | None,
    stance_foot_slip: torch.Tensor | None,
    helper: EnergySupportConsistencyLoss | None = None,
) -> torch.Tensor:
    physics = helper or _DEFAULT_ENERGY_SUPPORT
    return physics.support_consistency_loss(support_margin, stance_foot_slip)


def deformation_penalty_loss(delta_reference: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    if p == 1.0:
        return torch.abs(delta_reference).mean()
    return delta_reference.pow(2).mean()


def auxiliary_load_regression_loss(
    pred_load: torch.Tensor | None,
    target_load: torch.Tensor | None,
    *,
    enabled: bool = False,
) -> torch.Tensor:
    if not enabled or pred_load is None or target_load is None:
        device = pred_load.device if pred_load is not None else (target_load.device if target_load is not None else torch.device("cpu"))
        dtype = pred_load.dtype if pred_load is not None else (target_load.dtype if target_load is not None else torch.float32)
        return torch.zeros((), device=device, dtype=dtype)

    if target_load.dtype in {torch.int32, torch.int64, torch.long}:
        return F.cross_entropy(pred_load, target_load)
    return F.mse_loss(pred_load, target_load)


def uncertainty_regularization_loss(
    uncertainty: torch.Tensor | None,
    *,
    enabled: bool = False,
) -> torch.Tensor:
    if not enabled or uncertainty is None:
        if uncertainty is None:
            return torch.zeros(())
        return torch.zeros((), device=uncertainty.device, dtype=uncertainty.dtype)
    return uncertainty.pow(2).mean()


# Optional legacy hooks retained for future full design.
def energy_consistency_loss(
    state_t: torch.Tensor,
    state_tp1: torch.Tensor,
    action: torch.Tensor,
    external_wrench: torch.Tensor | None = None,
    helper: EnergySupportConsistencyLoss | None = None,
) -> torch.Tensor:
    physics = helper or _DEFAULT_ENERGY_SUPPORT
    return physics.energy_consistency_loss(state_t, state_tp1, action, external_wrench=external_wrench)


def optional_slow_latent_alignment_loss(
    z_slow: torch.Tensor,
    load_regime: torch.Tensor | None = None,
    *,
    enabled: bool = False,
    helper: EnergySupportConsistencyLoss | None = None,
) -> torch.Tensor:
    if not enabled:
        return torch.zeros((), device=z_slow.device, dtype=z_slow.dtype)
    physics = helper or _DEFAULT_ENERGY_SUPPORT
    return physics.slow_latent_alignment_loss(z_slow, load_regime=load_regime)
