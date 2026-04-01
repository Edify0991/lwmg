from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Observation:
    q: torch.Tensor
    dq: torch.Tensor
    imu_accel: torch.Tensor
    imu_gyro: torch.Tensor
    prev_action: torch.Tensor
    contacts: torch.Tensor
    tracking_error_summary: torch.Tensor
    root_quat_wxyz: torch.Tensor | None = None
    projected_gravity: torch.Tensor | None = None
