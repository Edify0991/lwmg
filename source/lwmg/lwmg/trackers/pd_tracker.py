from __future__ import annotations

import torch

from .base_tracker import BaseTracker


class PDTracker(BaseTracker):
    def __init__(self, kp: float = 60.0, kd: float = 4.0) -> None:
        self.kp = kp
        self.kd = kd

    def act(self, observation: torch.Tensor) -> torch.Tensor:
        n = observation.shape[-1] // 2
        q, dq = observation[..., :n], observation[..., n:]
        return -self.kp * q - self.kd * dq
