import torch

from lwmg.world_model.ensemble_wrapper import EnsembleWrapper


class _Dummy:
    def __init__(self, offset: float) -> None:
        self.offset = offset

    def rollout(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.offset


def test_uncertainty_rollout() -> None:
    ens = EnsembleWrapper([_Dummy(0.0), _Dummy(1.0), _Dummy(2.0)])
    mean, var = ens.rollout(torch.zeros(2, 3))
    assert mean.shape == var.shape == (2, 3)
    assert torch.all(var > 0)
