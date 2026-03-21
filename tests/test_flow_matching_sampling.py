import torch

from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator


def test_flow_matching_velocity_and_decode() -> None:
    gen = FlowMatchingGenerator()
    x = torch.randn(2, 4, gen.latent_dim)
    tau = torch.rand(2, 1)
    ctx = torch.randn(2, 16)
    v = gen.velocity_field(x, tau, ctx)
    assert v.shape == x.shape
    dec = gen.decode_reference(x)
    assert dec.shape[-1] == 29
