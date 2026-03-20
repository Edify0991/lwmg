import torch

from lwmg.world_model import losses


def test_losses_non_negative() -> None:
    x = torch.zeros(3, 4)
    y = torch.ones(3, 4)
    assert losses.state_prediction_loss(x, y) >= 0
    assert losses.rollout_loss(x, y) >= 0
