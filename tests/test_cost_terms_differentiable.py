import torch

from lwmg.guidance.cost_terms import smoothness_cost


def test_cost_term_differentiable() -> None:
    ref = torch.randn(2, 6, 29, requires_grad=True)
    loss = smoothness_cost(ref)
    loss.backward()
    assert ref.grad is not None
