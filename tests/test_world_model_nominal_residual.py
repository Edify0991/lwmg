import torch

from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def test_world_model_nominal_residual_composition() -> None:
    wm = StructuredClosedLoopWorldModel()
    s = torch.randn(3, 32)
    r = torch.randn(3, 29)
    u = torch.randn(3, 29)
    h = torch.randn(3, 4, 32)
    out = wm.step(s, r, u, h)
    assert out.shape == s.shape
