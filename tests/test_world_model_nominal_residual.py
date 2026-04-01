import torch

from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def test_world_model_nominal_residual_composition() -> None:
    wm = StructuredClosedLoopWorldModel()
    s = torch.randn(3, 32)
    r = torch.randn(3, 29)
    u = torch.randn(3, 29)
    h_slow = torch.randn(3, 8, 32)
    h_fast = torch.randn(3, 4, 32)
    z = wm.encode_interaction(h_slow, h_fast)
    assert z.shape == (3, wm.history_dims.fused)

    out, debug = wm.predict_step(s, r, u, h_slow, h_fast, return_details=True)
    assert out.shape == s.shape
    assert debug["delta_s_load"].shape == s.shape

    unc = wm.predict_uncertainty(s, torch.randn(3, 2, 29), torch.randn(3, 2, 29), h_slow, h_fast)
    assert unc.shape[0] == 3
