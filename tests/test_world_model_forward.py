import torch

from lwmg.world_model.wm_module import LoadAwareWorldModel


def test_world_model_forward_shapes() -> None:
    model = LoadAwareWorldModel(state_dim=16, hidden_dim=32, latent_dim=8)
    out = model(torch.randn(4, 16), torch.randn(4, 3, 16))
    assert out["next_state"].shape == (4, 16)
    assert out["hard_failure_logit"].shape == (4, 1)
