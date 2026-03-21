import torch

from lwmg.guidance.cost_terms import (
    prior_anchor_cost,
    smoothness_cost,
    stability_cost,
    task_cost,
    tracking_feasibility_cost,
    uncertainty_cost,
)


def test_cost_terms_differentiable() -> None:
    ref = torch.randn(2, 6, 29, requires_grad=True)
    target = torch.randn(2, requires_grad=True)
    cost = (
        task_cost(target, target * 0)
        + tracking_feasibility_cost(ref.mean(dim=-1))
        + stability_cost(ref[..., 0], ref[..., 1], ref[..., 2], ref[..., 3], ref[..., :6])
        + smoothness_cost(ref)
        + uncertainty_cost(ref.var(dim=-1, unbiased=False))
        + prior_anchor_cost(ref, torch.zeros_like(ref))
    )
    cost.backward()
    assert ref.grad is not None
