import torch

from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def test_world_model_forward_shapes() -> None:
    model = StructuredClosedLoopWorldModel(state_dim=16, ref_dim=8, ctrl_dim=8, latent_dim=4)
    s0 = torch.randn(4, 16)
    refs = torch.randn(4, 3, 8)
    ctrls = torch.randn(4, 3, 8)
    hist = torch.randn(4, 2, 16)
    out = model.rollout(s0, refs, ctrls, hist)
    assert out.shape == (4, 4, 16)
