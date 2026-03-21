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
            imu_accel=torch.zeros(3, dtype=q.dtype, device=q.device),
            imu_gyro=torch.zeros(3, dtype=q.dtype, device=q.device),
            prev_action=action,
            contacts=torch.tensor([1.0, 1.0], dtype=q.dtype, device=q.device),
            tracking_error_summary=tracking_error,
        )
        flags = check_terminations(q=q, dq=dq, tracking_error=tracking_error)
        return EnvStepOutput(observation=obs, privileged=privileged, flags=flags)


class LwmgVectorEnv:
    """Small vectorized env shim used by standalone scripts."""

    def __init__(self, num_envs: int, num_joints: int = 12, device: str = "cpu") -> None:
        self.num_envs = num_envs
        self.num_joints = num_joints
        self.device = torch.device(device)
        self.single = HumanoidLoadEnv(num_joints=num_joints)
        self.q = torch.zeros(num_envs, num_joints, device=self.device)
        self.dq = torch.zeros(num_envs, num_joints, device=self.device)
        self.prev_action = torch.zeros(num_envs, num_joints, device=self.device)

    def reset(self) -> dict[str, torch.Tensor]:
        self.q.zero_()
        self.dq.zero_()
        self.prev_action.zero_()
        return {"q": self.q.clone(), "dq": self.dq.clone(), "prev_action": self.prev_action.clone()}

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.q[env_ids] = 0.0
        self.dq[env_ids] = 0.0
        self.prev_action[env_ids] = 0.0

    def step(self, action: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        outputs: list[EnvStepOutput] = []
        for i in range(self.num_envs):
            out = self.single.step(self.q[i], self.dq[i], action[i])
            self.prev_action[i] = action[i]
            self.dq[i] = action[i] - self.q[i]
            self.q[i] = self.q[i] + 0.05 * self.dq[i]
            outputs.append(out)

        done = torch.tensor([o.flags.hard_failure for o in outputs], dtype=torch.bool, device=self.device)
        reward = -torch.stack([o.observation.tracking_error_summary.squeeze(0) for o in outputs]).to(self.device)
        obs = {"q": self.q.clone(), "dq": self.dq.clone(), "prev_action": self.prev_action.clone()}
        info = {
            "soft_failure": torch.tensor(
                [
                    o.flags.excessive_trunk_tilt_warning
                    or o.flags.slip_warning
                    or o.flags.persistent_high_torque_warning
                    or o.flags.growing_tracking_error_warning
                    for o in outputs
                ],
                dtype=torch.bool,
                device=self.device,
            )
        }
        return obs, reward, done, info


def make_lwmg_env(env_cfg: dict, device: str = "cpu") -> LwmgVectorEnv:
    num_envs = int(env_cfg.get("num_envs", 1))
    num_joints = int(env_cfg.get("num_joints", 12))
    return LwmgVectorEnv(num_envs=num_envs, num_joints=num_joints, device=device)
