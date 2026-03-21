from __future__ import annotations

import torch


def euler_integrate(ode_fn, x0: torch.Tensor, n_steps: int = 16) -> torch.Tensor:
    x = x0
    dt = 1.0 / max(1, n_steps)
    for i in range(n_steps):
        tau = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
        x = x + dt * ode_fn(tau, x)
    return x


def integrate_ode(ode_fn, x0: torch.Tensor, n_steps: int = 16, method: str = "euler") -> torch.Tensor:
    if method != "euler":
        try:
            from torchdiffeq import odeint  # type: ignore

            ts = torch.linspace(0.0, 1.0, steps=n_steps + 1, device=x0.device, dtype=x0.dtype)
            out = odeint(lambda t, x: ode_fn(torch.full((x.shape[0], 1), float(t), device=x.device, dtype=x.dtype), x), x0, ts)
            return out[-1]
        except Exception:
            return euler_integrate(ode_fn, x0, n_steps=n_steps)
    return euler_integrate(ode_fn, x0, n_steps=n_steps)
