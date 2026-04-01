from __future__ import annotations

import torch


def task_cost_balance(rollout_states: torch.Tensor, target_root_height: float = 0.78) -> torch.Tensor:
    root_z = rollout_states[..., 0]
    height_cost = (root_z - target_root_height).pow(2).mean()
    tilt_cost = rollout_states[..., 1:3].pow(2).mean() if rollout_states.shape[-1] >= 3 else torch.zeros((), device=rollout_states.device)
    return height_cost + 0.5 * tilt_cost


def task_cost_standup(
    rollout_states: torch.Tensor,
    target_root_height: float = 0.82,
    terminal_upright_target: float = 0.0,
) -> torch.Tensor:
    terminal = rollout_states[:, -1]
    z_terminal = terminal[..., 0]
    height = (z_terminal - target_root_height).pow(2).mean()
    upright = (terminal[..., 1] - terminal_upright_target).pow(2).mean() if terminal.shape[-1] > 1 else torch.zeros((), device=rollout_states.device)
    vel_pen = rollout_states[:, -1, 3:6].pow(2).mean() if rollout_states.shape[-1] >= 6 else torch.zeros((), device=rollout_states.device)
    return height + 0.5 * upright + 0.2 * vel_pen


def task_cost_locomotion(
    rollout_states: torch.Tensor,
    target_planar_vel: tuple[float, float] = (0.5, 0.0),
) -> torch.Tensor:
    if rollout_states.shape[-1] < 5:
        return torch.zeros((), device=rollout_states.device)
    vxy = rollout_states[..., 3:5]
    tgt = torch.tensor(target_planar_vel, device=rollout_states.device, dtype=rollout_states.dtype)
    return (vxy - tgt).pow(2).mean()


def tracking_cost(reference: torch.Tensor, adapted_reference: torch.Tensor) -> torch.Tensor:
    return (adapted_reference - reference).pow(2).mean()


def stability_cost(
    support_margin: torch.Tensor | None,
    stance_slip: torch.Tensor | None,
    margin_safe: float = 0.015,
    slip_safe: float = 0.10,
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    out_device = support_margin.device if support_margin is not None else (
        stance_slip.device if stance_slip is not None else (device or torch.device("cpu"))
    )
    out_dtype = support_margin.dtype if support_margin is not None else (
        stance_slip.dtype if stance_slip is not None else (dtype or torch.float32)
    )
    out = torch.zeros((), device=out_device, dtype=out_dtype)

    if support_margin is not None:
        out = out + torch.nn.functional.softplus((margin_safe - support_margin) * 10.0).mean()
    if stance_slip is not None:
        out = out + torch.relu(torch.abs(stance_slip) - slip_safe).pow(2).mean()
    return out


def smoothness_cost(reference: torch.Tensor) -> torch.Tensor:
    if reference.shape[1] <= 1:
        return torch.zeros((), device=reference.device, dtype=reference.dtype)
    vel = reference[:, 1:] - reference[:, :-1]
    if vel.shape[1] <= 1:
        return vel.pow(2).mean()
    acc = vel[:, 1:] - vel[:, :-1]
    return 0.5 * vel.pow(2).mean() + acc.pow(2).mean()


def deformation_cost(delta_reference: torch.Tensor) -> torch.Tensor:
    return delta_reference.pow(2).mean()


def compose_total_cost(
    *,
    task: torch.Tensor,
    tracking: torch.Tensor,
    stability: torch.Tensor,
    smoothness: torch.Tensor,
    deformation: torch.Tensor,
    uncertainty: torch.Tensor | None = None,
    weights: dict[str, float] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    w = {
        "task": 1.0,
        "tracking": 1.0,
        "stability": 1.0,
        "smoothness": 0.2,
        "deformation": 0.2,
        "uncertainty": 0.0,
    }
    if weights is not None:
        w.update({k: float(v) for k, v in weights.items()})

    unc = uncertainty if uncertainty is not None else torch.zeros_like(task)
    total = (
        w["task"] * task
        + w["tracking"] * tracking
        + w["stability"] * stability
        + w["smoothness"] * smoothness
        + w["deformation"] * deformation
        + w["uncertainty"] * unc
    )
    return total, {
        "task": task,
        "tracking": tracking,
        "stability": stability,
        "smoothness": smoothness,
        "deformation": deformation,
        "uncertainty": unc,
        "total": total,
    }
