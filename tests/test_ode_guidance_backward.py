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

    grad = guidance.grad_guidance(x, flow.decode_reference, anchor)
    assert grad.shape == x.shape

    tau = torch.rand(2, 1)
    vfn = lambda x_tau, tau_: flow.velocity_field(x_tau, tau_, torch.randn(2, 16))
    step = guidance.guided_step(x, tau, vfn, flow.decode_reference, anchor)
    assert step.shape == x.shape
