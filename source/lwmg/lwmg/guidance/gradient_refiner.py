from __future__ import annotations

import torch


class GradientRefiner:
    def __init__(self, steps: int = 3, lr: float = 0.05) -> None:
        self.steps = steps
        self.lr = lr

    def refine(self, x: torch.Tensor, cost_fn) -> torch.Tensor:
        y = x.clone().detach().requires_grad_(True)
        for _ in range(self.steps):
            cost = cost_fn(y).mean()
            grad = torch.autograd.grad(cost, y)[0]
            y = (y - self.lr * grad).detach().requires_grad_(True)
        return y.detach()
