from __future__ import annotations

import torch


def rollout_closed_loop(model, s0: torch.Tensor, references: torch.Tensor, controls: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
    states = [s0]
    s_t = s0
    for t in range(references.shape[1]):
        r_t = references[:, t]
        u_t = controls[:, t]
        s_t = model.step(s_t, r_t, u_t, history)
        states.append(s_t)
    return torch.stack(states, dim=1)
