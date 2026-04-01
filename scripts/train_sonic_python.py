# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""In-process SONIC ONNX rollout/training harness on LWMG Direct env (no DDS)."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
from pathlib import Path
from typing import Any

import yaml

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None


def _require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing module '{module_name}'. {install_hint}")


_require_module("isaaclab.app", "Install Isaac Lab and run this script with the Isaac Lab Python launcher.")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run frozen SONIC ONNX in-process inside IsaacLab (training/rollout mode).")
parser.add_argument("--env-config", type=Path, required=True, help="Path to env YAML config.")
parser.add_argument(
    "--tracker-config",
    type=Path,
    default=Path("configs/tracker/frozen_sonic_tracker.yaml"),
    help="Path to frozen SONIC tracker YAML.",
)
parser.add_argument(
    "--reference-clip",
    type=Path,
    default=None,
    help="Optional clip directory containing joint_pos.csv/joint_vel.csv/body_pos.csv/body_quat.csv.",
)
parser.add_argument(
    "--allow-degenerate-reference",
    action="store_true",
    help="Allow near-zero placeholder reference clips (not recommended).",
)
parser.add_argument("--num-envs", type=int, default=64, help="Number of vectorized environments.")
parser.add_argument("--max-steps", type=int, default=2000, help="Max rollout steps.")
parser.add_argument("--log-interval", type=int, default=50, help="Print stats every N steps.")
parser.add_argument("--diagnose", action="store_true", help="Enable online action/reference diagnostics.")
parser.add_argument("--diag-interval", type=int, default=100, help="Diagnostic print interval in steps.")
parser.add_argument("--trace-out", type=Path, default=None, help="Optional NPZ path to dump env0 trajectory traces.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from lwmg.references.reference_types import ReferenceTarget
from lwmg.references.replay_reference import ReplayReferenceGenerator
from lwmg.tasks.direct.lwmg.lwmg_env import LwmgEnv
from lwmg.tasks.direct.lwmg.lwmg_env_cfg import LwmgEnvCfg
from lwmg.trackers.sonic_python_runtime import SonicPythonRuntime

_OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+),\s*(.+)\}$")


def _resolve_oc_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _OC_ENV_PATTERN.match(value.strip())
    if match is None:
        return value
    env_name = match.group(1).strip()
    default = match.group(2).strip()
    if default.endswith("}"):
        default = default[:-1].strip()
    return os.environ.get(env_name, default)


def _resolve_path(path_value: Any, base_dir: Path) -> Path:
    raw = _resolve_oc_env(path_value)
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate
    by_cfg = (base_dir / candidate).resolve()
    if by_cfg.exists():
        return by_cfg
    return (Path.cwd() / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got type={type(data).__name__}")
    return data


def _build_direct_env(env_cfg: dict[str, Any], tracker_cfg: dict[str, Any], num_envs: int) -> LwmgEnv:
    cfg = LwmgEnvCfg()

    sim_cfg = env_cfg.get("sim", {})
    if "dt" in sim_cfg:
        cfg.sim.dt = float(sim_cfg["dt"])
    if "control_decimation" in sim_cfg:
        cfg.decimation = int(sim_cfg["control_decimation"])
    cfg.sim.render_interval = cfg.decimation

    scene_cfg = env_cfg.get("scene", {})
    cfg.scene.num_envs = int(num_envs)
    if "env_spacing" in scene_cfg:
        cfg.scene.env_spacing = float(scene_cfg["env_spacing"])

    robot_cfg = env_cfg.get("robot", {})
    if "spawn_height" in robot_cfg:
        cfg.robot_cfg.init_state.pos = (0.0, 0.0, float(robot_cfg["spawn_height"]))
    if "self_collisions" in robot_cfg:
        cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = bool(robot_cfg["self_collisions"])
    if "usd_path" in robot_cfg:
        cfg.robot_cfg.spawn.usd_path = str(_resolve_path(robot_cfg["usd_path"], Path.cwd()))

    action_cfg = tracker_cfg.get("action_mapping", {}) if isinstance(tracker_cfg.get("action_mapping", {}), dict) else {}

    # Python runtime outputs absolute q_target reconstructed as: default_angles + raw * g1_action_scale.
    cfg.control_mode = "joint_target_absolute"
    cfg.action_scale = 1.0
    cfg.clamp_to_joint_limits = bool(action_cfg.get("clamp_to_joint_limits", True))

    return LwmgEnv(cfg)


def _build_reference_generator(
    env_cfg: dict[str, Any],
    explicit_clip: Path | None,
    joint_dim: int,
    device: str,
    allow_degenerate_clip: bool,
) -> ReplayReferenceGenerator | None:
    ref_cfg = env_cfg.get("reference", {})

    clip_dir = explicit_clip
    if clip_dir is None:
        clip_dir_raw = ref_cfg.get("clip_dir", None)
        if clip_dir_raw is not None:
            clip_dir = _resolve_path(clip_dir_raw, Path.cwd())

    if clip_dir is None:
        return None

    loop = bool(ref_cfg.get("loop", True))
    allow_degenerate = bool(ref_cfg.get("allow_degenerate_clip", allow_degenerate_clip))

    return ReplayReferenceGenerator(
        clip_dir=clip_dir,
        joint_dim=joint_dim,
        device=device,
        loop=loop,
        allow_degenerate_clip=allow_degenerate,
    )


def _slice_reference(reference: ReferenceTarget | None, env_id: int) -> ReferenceTarget | None:
    if reference is None:
        return None

    def _slice(t: torch.Tensor | None) -> torch.Tensor | None:
        if t is None:
            return None
        if t.ndim == 0:
            return t
        return t[env_id : env_id + 1]

    extras: dict[str, torch.Tensor] = {}
    for key, value in reference.extras.items():
        if not isinstance(value, torch.Tensor):
            continue
        if value.ndim > 0 and value.shape[0] == reference.joint_pos.shape[0]:
            extras[key] = value[env_id : env_id + 1]
        else:
            extras[key] = value

    return ReferenceTarget(
        joint_pos=reference.joint_pos[env_id : env_id + 1],
        joint_vel=reference.joint_vel[env_id : env_id + 1],
        root_pos=_slice(reference.root_pos),
        root_quat=_slice(reference.root_quat),
        frame_idx=_slice(reference.frame_idx),
        extras=extras,
    )


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().to(dtype=torch.float32).flatten()
    if flat.numel() == 0:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "abs_max": 0.0}
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_max": float(flat.abs().max().item()),
    }


def _format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
        f"min={stats['min']:.4f} max={stats['max']:.4f} |abs|max={stats['abs_max']:.4f}"
    )


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def main() -> None:
    env_cfg_all = _load_yaml(args_cli.env_config)
    tracker_cfg_all = _load_yaml(args_cli.tracker_config)

    direct_env = _build_direct_env(env_cfg_all, tracker_cfg=tracker_cfg_all, num_envs=args_cli.num_envs)
    runtime = SonicPythonRuntime.from_tracker_config(
        args_cli.tracker_config,
        target_dim=direct_env.num_joints,
        fallback_dir=Path.cwd(),
    )

    replay_gen = _build_reference_generator(
        env_cfg=env_cfg_all,
        explicit_clip=args_cli.reference_clip,
        joint_dim=direct_env.num_joints,
        device=str(direct_env.device),
        allow_degenerate_clip=args_cli.allow_degenerate_reference,
    )

    print(
        f"[train_sonic_python] num_envs={direct_env.num_envs} sim_dt={float(direct_env.cfg.sim.dt):.6f}s "
        f"decimation={int(direct_env.cfg.decimation)} control_dt={float(direct_env.cfg.sim.dt)*float(direct_env.cfg.decimation):.6f}s"
    )
    print(
        f"[train_sonic_python] provider={runtime.cfg.provider} joint_order={runtime.cfg.joint_order} "
        f"batch_mode={runtime.cfg.batch_mode} "
        f"clamp_raw_action={runtime.cfg.clamp_raw_action} action_clip={runtime.cfg.action_clip:.4f}"
    )

    if replay_gen is None:
        print("[train_sonic_python] reference: disabled (zero motion features for motion_* entries)")
    else:
        q_abs_max = float(torch.max(torch.abs(replay_gen.joint_pos)).item())
        dq_abs_max = float(torch.max(torch.abs(replay_gen.joint_vel)).item())
        root_z_min = float(torch.min(replay_gen.body_pos[:, 2]).item())
        root_z_max = float(torch.max(replay_gen.body_pos[:, 2]).item())
        print(
            f"[train_sonic_python] reference frames={replay_gen.num_frames} q_abs_max={q_abs_max:.4f} "
            f"dq_abs_max={dq_abs_max:.4f} root_z_range=[{root_z_min:.4f}, {root_z_max:.4f}]"
        )

    reset_out = direct_env.reset()
    _ = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    runtime.reset_all(direct_env.num_envs)

    trace_enabled = args_cli.trace_out is not None
    if trace_enabled and np is None:
        raise RuntimeError("Trace export requires numpy. Please install numpy in the runtime environment.")

    trace: dict[str, list[Any]] = {
        "step": [],
        "reward": [],
        "done": [],
        "frame_idx": [],
        "q_meas_abs": [],
        "raw_action": [],
        "q_target_abs": [],
        "q_ref_abs": [],
    }

    step = 0
    env_ids = list(range(direct_env.num_envs))
    while simulation_app.is_running() and step < args_cli.max_steps:
        with torch.inference_mode():
            ref_batch = replay_gen.generate(direct_env.num_envs) if replay_gen is not None else None
            observations = direct_env.get_tracker_observations()
            ref_list = [_slice_reference(ref_batch, env_id) for env_id in env_ids]

            frame_idx_env0 = -1
            q_ref_abs_env0 = torch.zeros((direct_env.num_joints,), dtype=torch.float32, device=direct_env.device)
            raw_actions, q_target_abs = runtime.infer_batch(
                currents=observations,
                references=ref_list,
                reference_generator=replay_gen,
                env_ids=env_ids,
            )

            ref_env0 = ref_list[0] if ref_list else None
            if ref_env0 is not None:
                if ref_env0.frame_idx is not None and ref_env0.frame_idx.numel() > 0:
                    frame_idx_env0 = int(ref_env0.frame_idx.flatten()[0].item())
                q_ref_abs_env0 = ref_env0.joint_pos[0].to(device=direct_env.device, dtype=torch.float32)

            # Keep SONIC last_actions aligned with previous raw policy output.
            direct_env.set_tracker_prev_action(raw_actions)

            step_out = direct_env.step(q_target_abs)
            if isinstance(step_out, tuple) and len(step_out) == 5:
                _obs, reward, terminated, truncated, _info = step_out
                done = terminated | truncated
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                _obs, reward, done, _info = step_out
            else:
                raise RuntimeError("Unexpected step() return structure from direct environment")

            done_env_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
            for env_id in done_env_ids:
                runtime.reset_env(int(env_id))

            q_meas_abs_env0, _ = direct_env.get_joint_state_abs_vel(env_id=0)

        if trace_enabled:
            trace["step"].append(step)
            trace["reward"].append(float(reward.mean().item()))
            trace["done"].append(int(done[0].item()))
            trace["frame_idx"].append(frame_idx_env0)
            trace["q_meas_abs"].append(_to_numpy(q_meas_abs_env0))
            trace["raw_action"].append(_to_numpy(raw_actions[0]))
            trace["q_target_abs"].append(_to_numpy(q_target_abs[0]))
            trace["q_ref_abs"].append(_to_numpy(q_ref_abs_env0))

        if step % max(1, args_cli.log_interval) == 0:
            print(
                f"[train_sonic_python] step={step} reward={reward.mean().item():.4f} "
                f"done_count={int(done.sum().item())} soft_fail={int(direct_env.soft_failure.sum().item())}"
            )

        if args_cli.diagnose and step % max(1, args_cli.diag_interval) == 0:
            raw_stats = _tensor_stats(raw_actions)
            target_stats = _tensor_stats(q_target_abs)
            ref_err_stats = _tensor_stats(q_ref_abs_env0 - q_meas_abs_env0)
            print(f"[train_sonic_python_diag] {_format_stats('raw_action', raw_stats)}")
            print(f"[train_sonic_python_diag] {_format_stats('q_target_abs', target_stats)}")
            print(f"[train_sonic_python_diag] {_format_stats('q_ref-q_meas(abs)_env0', ref_err_stats)}")

        step += 1

    if trace_enabled and args_cli.trace_out is not None:
        arrays: dict[str, np.ndarray] = {}
        for key, values in trace.items():
            if not values:
                continue
            first = values[0]
            if isinstance(first, np.ndarray):
                arrays[key] = np.stack(values, axis=0)
            else:
                arrays[key] = np.asarray(values)

        arrays["meta_num_envs"] = np.asarray(direct_env.num_envs)
        arrays["meta_sim_dt"] = np.asarray(float(direct_env.cfg.sim.dt))
        arrays["meta_decimation"] = np.asarray(int(direct_env.cfg.decimation))
        arrays["meta_control_dt"] = np.asarray(float(direct_env.cfg.sim.dt) * float(direct_env.cfg.decimation))
        arrays["meta_provider"] = np.asarray(runtime.cfg.provider)
        arrays["meta_batch_mode"] = np.asarray(int(runtime.cfg.batch_mode))
        arrays["meta_joint_order"] = np.asarray(runtime.cfg.joint_order)
        arrays["meta_tracker_config"] = np.asarray(str(args_cli.tracker_config.resolve()))

        args_cli.trace_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args_cli.trace_out, **arrays)
        print(f"[train_sonic_python] trace saved: {args_cli.trace_out}")

    direct_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
