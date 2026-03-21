import torch

from lwmg.guidance.ode_guidance import FlowODEGuidance
from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator
from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def test_ode_guidance_backward() -> None:
    flow = FlowMatchingGenerator()
    wm = StructuredClosedLoopWorldModel()
    guidance = FlowODEGuidance(wm)
    x = torch.randn(2, 5, flow.latent_dim)
    anchor = flow.decode_reference(torch.zeros_like(x))
    grad = guidance.gradient(x, flow.decode_reference, anchor)
    assert grad.shape == x.shape
