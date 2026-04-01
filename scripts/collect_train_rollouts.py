# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Collect batched training rollouts with in-process frozen SONIC ONNX on IsaacLab."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
from datetime import datetime
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


parser = argparse.ArgumentParser(
    description="Collect parallel training rollouts with frozen SONIC ONNX in-process (no DDS)."
)
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
parser.add_argument("--num-steps", type=int, default=4096, help="Number of control steps to collect.")
parser.add_argument(
    "--out-dir",
    type=Path,
    default=Path("outputs/train_rollouts"),
    help="Output root directory. A timestamped run directory is created inside.",
)
parser.add_argument(
    "--shard-steps",
    type=int,
    default=512,
    help="Steps per shard file (npz).",
)
parser.add_argument(
    "--include-sonic-obs",
    action="store_true",
    help="Also store SONIC decoder_obs/encoder_obs vectors per env per step (larger files).",
)
parser.add_argument("--log-interval", type=int, default=50, help="Print stats every N steps.")
parser.add_argument("--diag-interval", type=int, default=200, help="Print diagnostics every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from lwmg.envs.observations import Observation
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


def _build_direct_env(
    env_cfg: dict[str, Any],
    tracker_cfg: dict[str, Any],
    num_envs: int,
    env_cfg_dir: Path,
) -> LwmgEnv:
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
        cfg.robot_cfg.spawn.usd_path = str(_resolve_path(robot_cfg["usd_path"], env_cfg_dir))

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
    env_cfg_dir: Path,
) -> ReplayReferenceGenerator | None:
    ref_cfg = env_cfg.get("reference", {})

    clip_dir = explicit_clip
    if clip_dir is not None and not clip_dir.is_absolute():
        clip_dir = (Path.cwd() / clip_dir).resolve()
    if clip_dir is None:
        clip_dir_raw = ref_cfg.get("clip_dir", None)
        if clip_dir_raw is not None:
            clip_dir = _resolve_path(clip_dir_raw, env_cfg_dir)

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


def _stack_obs_fields(observations: list[Observation]) -> dict[str, torch.Tensor]:
    return {
        "q": torch.stack([o.q for o in observations], dim=0),
        "dq": torch.stack([o.dq for o in observations], dim=0),
        "imu_gyro": torch.stack([o.imu_gyro for o in observations], dim=0),
        "prev_action": torch.stack([o.prev_action for o in observations], dim=0),
        "tracking_error_summary": torch.stack([o.tracking_error_summary for o in observations], dim=0),
        "root_quat_wxyz": torch.stack(
            [o.root_quat_wxyz if o.root_quat_wxyz is not None else torch.tensor([1.0, 0.0, 0.0, 0.0], device=o.q.device) for o in observations],
            dim=0,
        ),
        "projected_gravity": torch.stack(
            [
                o.projected_gravity
                if o.projected_gravity is not None
                else torch.tensor([0.0, 0.0, -1.0], device=o.q.device)
                for o in observations
            ],
            dim=0,
        ),
    }


def _zeros_reference_batch(num_envs: int, num_joints: int, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "joint_pos": torch.zeros((num_envs, num_joints), dtype=torch.float32, device=device),
        "joint_vel": torch.zeros((num_envs, num_joints), dtype=torch.float32, device=device),
        "root_pos": torch.zeros((num_envs, 3), dtype=torch.float32, device=device),
        "root_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=device).repeat(num_envs, 1),
        "frame_idx": torch.full((num_envs,), -1, dtype=torch.long, device=device),
    }


def _to_numpy(t: torch.Tensor) -> np.ndarray:
    assert np is not None
    return t.detach().cpu().numpy()


def _init_store(include_sonic_obs: bool) -> dict[str, list[np.ndarray]]:
    store: dict[str, list[np.ndarray]] = {
        "obs_q": [],
        "obs_dq": [],
        "obs_imu_gyro": [],
        "obs_prev_action": [],
        "obs_root_quat_wxyz": [],
        "obs_projected_gravity": [],
        "obs_tracking_error_summary": [],
        "next_obs_q": [],
        "next_obs_dq": [],
        "next_obs_imu_gyro": [],
        "next_obs_prev_action": [],
        "next_obs_root_quat_wxyz": [],
        "next_obs_projected_gravity": [],
        "next_obs_tracking_error_summary": [],
        "ref_joint_pos": [],
        "ref_joint_vel": [],
        "ref_root_pos": [],
        "ref_root_quat": [],
        "ref_frame_idx": [],
        "sonic_raw_action_isaac": [],
        "target_q_abs": [],
        "reward": [],
        "done": [],
        "soft_failure": [],
        "clamped_fraction": [],
    }
    if include_sonic_obs:
        store["sonic_decoder_obs"] = []
        store["sonic_encoder_obs"] = []
    return store


def _append_step(
    *,
    store: dict[str, list[np.ndarray]],
    obs_pre: dict[str, torch.Tensor],
    obs_post: dict[str, torch.Tensor],
    ref_batch: dict[str, torch.Tensor],
    raw_actions: torch.Tensor,
    q_target_abs: torch.Tensor,
    reward: torch.Tensor,
    done: torch.Tensor,
    soft_failure: torch.Tensor,
    clamped_fraction: torch.Tensor,
    decoder_obs: torch.Tensor | None,
    encoder_obs: torch.Tensor | None,
) -> None:
    store["obs_q"].append(_to_numpy(obs_pre["q"]))
    store["obs_dq"].append(_to_numpy(obs_pre["dq"]))
    store["obs_imu_gyro"].append(_to_numpy(obs_pre["imu_gyro"]))
    store["obs_prev_action"].append(_to_numpy(obs_pre["prev_action"]))
    store["obs_root_quat_wxyz"].append(_to_numpy(obs_pre["root_quat_wxyz"]))
    store["obs_projected_gravity"].append(_to_numpy(obs_pre["projected_gravity"]))
    store["obs_tracking_error_summary"].append(_to_numpy(obs_pre["tracking_error_summary"]))

    store["next_obs_q"].append(_to_numpy(obs_post["q"]))
    store["next_obs_dq"].append(_to_numpy(obs_post["dq"]))
    store["next_obs_imu_gyro"].append(_to_numpy(obs_post["imu_gyro"]))
    store["next_obs_prev_action"].append(_to_numpy(obs_post["prev_action"]))
    store["next_obs_root_quat_wxyz"].append(_to_numpy(obs_post["root_quat_wxyz"]))
    store["next_obs_projected_gravity"].append(_to_numpy(obs_post["projected_gravity"]))
    store["next_obs_tracking_error_summary"].append(_to_numpy(obs_post["tracking_error_summary"]))

    store["ref_joint_pos"].append(_to_numpy(ref_batch["joint_pos"]))
    store["ref_joint_vel"].append(_to_numpy(ref_batch["joint_vel"]))
    store["ref_root_pos"].append(_to_numpy(ref_batch["root_pos"]))
    store["ref_root_quat"].append(_to_numpy(ref_batch["root_quat"]))
    store["ref_frame_idx"].append(_to_numpy(ref_batch["frame_idx"]))

    store["sonic_raw_action_isaac"].append(_to_numpy(raw_actions))
    store["target_q_abs"].append(_to_numpy(q_target_abs))
    store["reward"].append(_to_numpy(reward))
    store["done"].append(_to_numpy(done.to(dtype=torch.int32)))
    store["soft_failure"].append(_to_numpy(soft_failure.to(dtype=torch.int32)))
    store["clamped_fraction"].append(_to_numpy(clamped_fraction))

    if decoder_obs is not None and "sonic_decoder_obs" in store:
        store["sonic_decoder_obs"].append(_to_numpy(decoder_obs))
    if encoder_obs is not None and "sonic_encoder_obs" in store:
        store["sonic_encoder_obs"].append(_to_numpy(encoder_obs))


def _flush_shard(
    *,
    run_dir: Path,
    shard_idx: int,
    step_start: int,
    step_end_exclusive: int,
    store: dict[str, list[np.ndarray]],
    meta_common: dict[str, Any],
) -> None:
    assert np is not None
    shard_path = run_dir / f"rollout_shard_{shard_idx:05d}.npz"

    arrays: dict[str, np.ndarray] = {}
    for key, values in store.items():
        if not values:
            continue
        arrays[key] = np.stack(values, axis=0)

    arrays["meta_step_start"] = np.asarray(step_start, dtype=np.int64)
    arrays["meta_step_end_exclusive"] = np.asarray(step_end_exclusive, dtype=np.int64)
    arrays["meta_num_steps"] = np.asarray(step_end_exclusive - step_start, dtype=np.int64)
    for k, v in meta_common.items():
        arrays[f"meta_{k}"] = np.asarray(v)

    np.savez_compressed(shard_path, **arrays)
    print(
        f"[collect_train_rollouts] saved shard={shard_idx:05d} "
        f"steps=[{step_start}, {step_end_exclusive}) path={shard_path}"
    )


def main() -> None:
    if np is None:
        raise RuntimeError("numpy is required for rollout collection. Please install numpy in this runtime.")

    env_cfg_path = args_cli.env_config.resolve()
    tracker_cfg_path = args_cli.tracker_config.resolve()
    env_cfg_all = _load_yaml(env_cfg_path)
    tracker_cfg_all = _load_yaml(tracker_cfg_path)

    direct_env = _build_direct_env(
        env_cfg_all,
        tracker_cfg=tracker_cfg_all,
        num_envs=args_cli.num_envs,
        env_cfg_dir=env_cfg_path.parent,
    )
    runtime = SonicPythonRuntime.from_tracker_config(
        tracker_cfg_path,
        target_dim=direct_env.num_joints,
        fallback_dir=Path.cwd(),
    )

    replay_gen = _build_reference_generator(
        env_cfg=env_cfg_all,
        explicit_clip=args_cli.reference_clip,
        joint_dim=direct_env.num_joints,
        device=str(direct_env.device),
        allow_degenerate_clip=args_cli.allow_degenerate_reference,
        env_cfg_dir=env_cfg_path.parent,
    )

    sim_dt = float(direct_env.cfg.sim.dt)
    decimation = int(direct_env.cfg.decimation)
    control_dt = sim_dt * float(decimation)

    print(
        f"[collect_train_rollouts] num_envs={direct_env.num_envs} sim_dt={sim_dt:.6f}s "
        f"decimation={decimation} control_dt={control_dt:.6f}s"
    )
    print(
        f"[collect_train_rollouts] provider={runtime.cfg.provider} joint_order={runtime.cfg.joint_order} "
        f"include_sonic_obs={bool(args_cli.include_sonic_obs)}"
    )

    if replay_gen is None:
        print("[collect_train_rollouts] reference: disabled (zero reference fields)")
    else:
        print(f"[collect_train_rollouts] reference_frames={replay_gen.num_frames} loop={bool(replay_gen.loop)}")

    run_stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = args_cli.out_dir.resolve() / f"sonic_rollout_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    meta_json = {
        "env_config": str(env_cfg_path),
        "tracker_config": str(tracker_cfg_path),
        "reference_clip": "" if args_cli.reference_clip is None else str(args_cli.reference_clip.resolve()),
        "num_envs": int(direct_env.num_envs),
        "num_steps": int(args_cli.num_steps),
        "shard_steps": int(args_cli.shard_steps),
        "include_sonic_obs": bool(args_cli.include_sonic_obs),
        "sim_dt": sim_dt,
        "decimation": decimation,
        "control_dt": control_dt,
        "provider": runtime.cfg.provider,
        "joint_order": runtime.cfg.joint_order,
        "created_at": run_stamp,
    }
    (run_dir / "meta.json").write_text(json.dumps(meta_json, indent=2, sort_keys=True))

    reset_out = direct_env.reset()
    _ = reset_out[0] if isinstance(reset_out, tuple) else reset_out
    runtime.reset_all(direct_env.num_envs)

    store = _init_store(include_sonic_obs=bool(args_cli.include_sonic_obs))
    shard_idx = 0
    shard_step_start = 0

    step = 0
    while simulation_app.is_running() and step < args_cli.num_steps:
        with torch.inference_mode():
            ref_target = replay_gen.generate(direct_env.num_envs) if replay_gen is not None else None
            obs_pre_list = direct_env.get_tracker_observations()
            obs_pre = _stack_obs_fields(obs_pre_list)

            if ref_target is None:
                ref_batch = _zeros_reference_batch(direct_env.num_envs, direct_env.num_joints, direct_env.device)
            else:
                ref_batch = {
                    "joint_pos": ref_target.joint_pos,
                    "joint_vel": ref_target.joint_vel,
                    "root_pos": ref_target.root_pos
                    if ref_target.root_pos is not None
                    else torch.zeros((direct_env.num_envs, 3), dtype=torch.float32, device=direct_env.device),
                    "root_quat": ref_target.root_quat
                    if ref_target.root_quat is not None
                    else torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=direct_env.device).repeat(
                        direct_env.num_envs, 1
                    ),
                    "frame_idx": ref_target.frame_idx
                    if ref_target.frame_idx is not None
                    else torch.full((direct_env.num_envs,), -1, dtype=torch.long, device=direct_env.device),
                }

            raw_rows: list[torch.Tensor] = []
            target_rows: list[torch.Tensor] = []
            decoder_rows: list[torch.Tensor] = []
            encoder_rows: list[torch.Tensor] = []

            for env_id, obs in enumerate(obs_pre_list):
                ref_env = _slice_reference(ref_target, env_id)
                raw_isaac, q_target_abs, debug = runtime.infer_one(
                    current=obs,
                    reference=ref_env,
                    reference_generator=replay_gen,
                    env_id=env_id,
                    return_debug=bool(args_cli.include_sonic_obs),
                )
                raw_rows.append(raw_isaac)
                target_rows.append(q_target_abs)

                if args_cli.include_sonic_obs:
                    assert debug is not None
                    decoder_rows.append(debug["decoder_obs"].to(dtype=torch.float32, device=direct_env.device))
                    encoder_obs = debug.get("encoder_obs", None)
                    if encoder_obs is None:
                        encoder_rows.append(torch.zeros((0,), dtype=torch.float32, device=direct_env.device))
                    else:
                        encoder_rows.append(encoder_obs.to(dtype=torch.float32, device=direct_env.device))

            raw_actions = torch.stack(raw_rows, dim=0)
            q_target_abs = torch.stack(target_rows, dim=0)
            decoder_obs = torch.stack(decoder_rows, dim=0) if decoder_rows else None
            encoder_obs = torch.stack(encoder_rows, dim=0) if encoder_rows else None

            # Keep SONIC history feature last_actions aligned with previous raw output.
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

            obs_post = _stack_obs_fields(direct_env.get_tracker_observations())

            _append_step(
                store=store,
                obs_pre=obs_pre,
                obs_post=obs_post,
                ref_batch=ref_batch,
                raw_actions=raw_actions,
                q_target_abs=q_target_abs,
                reward=reward,
                done=done,
                soft_failure=direct_env.soft_failure,
                clamped_fraction=direct_env.last_clamped_fraction,
                decoder_obs=decoder_obs,
                encoder_obs=encoder_obs,
            )

        if step % max(1, args_cli.log_interval) == 0:
            print(
                f"[collect_train_rollouts] step={step}/{args_cli.num_steps} "
                f"reward_mean={reward.mean().item():.4f} done_count={int(done.sum().item())} "
                f"soft_fail={int(direct_env.soft_failure.sum().item())}"
            )

        if step % max(1, args_cli.diag_interval) == 0:
            print(
                f"[collect_train_rollouts_diag] action_abs_max={float(torch.max(torch.abs(raw_actions)).item()):.4f} "
                f"target_abs_max={float(torch.max(torch.abs(q_target_abs)).item()):.4f} "
                f"clamped_mean={float(direct_env.last_clamped_fraction.mean().item()):.4f}"
            )

        step += 1

        shard_len = len(store["reward"])
        if shard_len >= max(1, int(args_cli.shard_steps)) or step >= args_cli.num_steps:
            _flush_shard(
                run_dir=run_dir,
                shard_idx=shard_idx,
                step_start=shard_step_start,
                step_end_exclusive=step,
                store=store,
                meta_common={
                    "num_envs": direct_env.num_envs,
                    "sim_dt": sim_dt,
                    "decimation": decimation,
                    "control_dt": control_dt,
                    "provider": runtime.cfg.provider,
                    "joint_order": runtime.cfg.joint_order,
                    "include_sonic_obs": int(bool(args_cli.include_sonic_obs)),
                },
            )
            shard_idx += 1
            shard_step_start = step
            store = _init_store(include_sonic_obs=bool(args_cli.include_sonic_obs))

    print(f"[collect_train_rollouts] completed. run_dir={run_dir}")
    direct_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
