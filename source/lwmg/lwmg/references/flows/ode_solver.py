from __future__ import annotations

import torch


def euler_integrate(ode_fn, x0: torch.Tensor, n_steps: int = 16) -> torch.Tensor:
    x = x0
    dt = 1.0 / max(1, n_steps)
    for i in range(n_steps):
        tau = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * ode_fn(tau, x)
    return x
