from __future__ import annotations

from dataclasses import dataclass

import torch

from .observations import Observation
from .random_load_sampler import PrivilegedLoadLabel, RandomLoadSampler
from .terminations import FailureFlags, check_terminations


@dataclass
class EnvStepOutput:
    observation: Observation
    privileged: PrivilegedLoadLabel
    flags: FailureFlags


class HumanoidLoadEnv:
    """Minimal Isaac Lab-side abstraction for load randomized G1 rollouts."""

    def __init__(self, num_joints: int = 12) -> None:
        self.num_joints = num_joints
        self.sampler = RandomLoadSampler()

    def step(self, q: torch.Tensor, dq: torch.Tensor, action: torch.Tensor) -> EnvStepOutput:
        privileged = self.sampler.sample()
        tracking_error = torch.abs(action - q).mean().unsqueeze(0)
        obs = Observation(
            q=q,
            dq=dq,
            imu_accel=torch.zeros(3),
            imu_gyro=torch.zeros(3),
            prev_action=action,
            contacts=torch.tensor([1.0, 1.0]),
            tracking_error_summary=tracking_error,
        )
        flags = check_terminations(q=q, dq=dq, tracking_error=tracking_error)
        return EnvStepOutput(observation=obs, privileged=privileged, flags=flags)
