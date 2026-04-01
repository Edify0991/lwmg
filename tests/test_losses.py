import torch

from lwmg.world_model import losses


def test_losses_non_negative() -> None:
    x = torch.zeros(3, 4)
    y = torch.ones(3, 4)
    assert losses.nominal_state_loss(x, y) >= 0
    assert losses.residual_rollout_loss(x, y) >= 0
    assert losses.residual_zero_loss(x) >= 0
    assert losses.deformation_penalty_loss(y - x) >= 0


def test_support_consistency_loss_finite() -> None:
    margin = torch.rand(5) * 0.03
    slip = torch.randn(5) * 0.04
    out = losses.support_consistency_loss(margin, slip)
    assert torch.isfinite(out)
