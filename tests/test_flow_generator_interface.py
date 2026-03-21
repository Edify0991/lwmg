import torch

from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator


def test_flow_generator_interface() -> None:
    gen = FlowMatchingGenerator()
    ctx = torch.randn(2, 16)
    ref = gen.sample_unguided(2, 5, ctx)
    assert ref.shape == (2, 5, 29)

    x = torch.randn(2, 5, gen.latent_dim)
    tau = torch.rand(2, 1)
    loss = gen.training_loss(x, tau, ctx, torch.zeros_like(x))
    assert loss.ndim == 0
