from __future__ import annotations

import torch


def short_rollout(initial_state: torch.Tensor, transition_fn, steps: int = 3) -> torch.Tensor:
    states = [initial_state]
    state = initial_state
    for _ in range(steps):
        state = transition_fn(state)
        states.append(state)
    return torch.stack(states, dim=1)
