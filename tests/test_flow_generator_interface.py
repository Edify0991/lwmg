import torch

from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator


def test_flow_generator_interface() -> None:
    gen = FlowMatchingGenerator()
    ctx = torch.randn(2, 16)
    ref = gen.sample_unguided(2, 5, ctx)
    assert ref.shape == (2, 5, 29)
