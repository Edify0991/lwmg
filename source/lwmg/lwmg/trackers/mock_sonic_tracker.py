from __future__ import annotations

import torch

from .base_tracker import BaseTracker


class MockSonicTracker(BaseTracker):
    def __init__(self, gain: float = 0.2) -> None:
        self.gain = gain

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        return -self.gain * observation[..., : observation.shape[-1] // 2]
