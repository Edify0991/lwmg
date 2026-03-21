import torch

from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def test_world_model_nominal_residual_composition() -> None:
    wm = StructuredClosedLoopWorldModel()
    s = torch.randn(3, 32)
    r = torch.randn(3, 29)
    u = torch.randn(3, 29)
    h = torch.randn(3, 4, 32)
    z = wm.encode_interaction(h, r, wm.nominal(s, r, u))
    assert z.shape[0] == 3

    out = wm.predict_step(s, r, u, h)
    assert out.shape == s.shape

    unc = wm.predict_uncertainty(s, torch.randn(3, 2, 29), torch.randn(3, 2, 29), h)
    assert unc.shape[0] == 3
