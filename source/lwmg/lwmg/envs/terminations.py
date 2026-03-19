from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class FailureFlags:
    excessive_trunk_tilt_warning: bool = False
    slip_warning: bool = False
    persistent_high_torque_warning: bool = False
    growing_tracking_error_warning: bool = False
    base_height_crash: bool = False
    base_orientation_crash: bool = False
    numeric_divergence: bool = False
    unrecoverable_collapse: bool = False

    @property
    def hard_failure(self) -> bool:
        return any(
            [
                self.base_height_crash,
                self.base_orientation_crash,
                self.numeric_divergence,
                self.unrecoverable_collapse,
            ]
        )


def check_terminations(q: torch.Tensor, dq: torch.Tensor, tracking_error: torch.Tensor) -> FailureFlags:
    flags = FailureFlags()
    flags.excessive_trunk_tilt_warning = bool(torch.abs(q[0]) > 1.2)
    flags.slip_warning = bool(torch.abs(dq).max() > 10.0)
    flags.growing_tracking_error_warning = bool(tracking_error.item() > 0.5)
    flags.persistent_high_torque_warning = bool(torch.abs(dq).mean() > 5.0)
    flags.numeric_divergence = bool(torch.isnan(q).any() or torch.isnan(dq).any())
    flags.base_height_crash = bool(q.numel() > 1 and q[1] < 0.2)
    flags.base_orientation_crash = bool(torch.abs(q[0]) > 2.5)
    flags.unrecoverable_collapse = flags.base_height_crash and flags.base_orientation_crash
    return flags
