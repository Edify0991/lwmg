# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay motions on LWMG Direct env using official SONIC C++ inference via DDS bridge."""

from __future__ import annotations

import argparse
import importlib.util
import os
import re
import time
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


parser = argparse.ArgumentParser(description="Replay LWMG motion references with SONIC C++ deploy through DDS bridge.")
parser.add_argument("--env-config", type=Path, required=True, help="Path to env YAML config.")
parser.add_argument(
    "--tracker-config",
    type=Path,
    default=Path("configs/tracker/sonic_dds_bridge.yaml"),
    help="Path to DDS bridge YAML config.",
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
parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments. DDS mode supports 1.")
parser.add_argument("--max-steps", type=int, default=500, help="Max replay steps.")
parser.add_argument("--log-interval", type=int, default=50, help="Print stats every N steps.")
parser.add_argument("--diagnose", action="store_true", help="Enable online command/state diagnostics.")
parser.add_argument("--diag-interval", type=int, default=50, help="Diagnostic print interval in steps.")
parser.add_argument("--trace-out", type=Path, default=None, help="Optional NPZ path to dump per-frame replay traces.")
parser.add_argument(
    "--step-mode",
    type=str,
    choices=("physics", "rl"),
    default="physics",
    help="physics: publish/pull DDS every physics tick (official-like). rl: legacy one-step-per-control-tick path.",
)
parser.add_argument(
    "--report-rl-signals",
    action="store_true",
    help="In physics step-mode, also compute reward/done signals from DirectRLEnv logic for diagnostics.",
)
parser.add_argument(
    "--stream-reference",
    action="store_true",
    help="Publish reference clip frames to C++ ZMQManager pose topic (STREAMED_MOTION).",
)
parser.add_argument(
    "--strict-wait-dds",
    action="store_true",
    help="Do not advance IsaacLab physics until first DDS command has been received.",
)
parser.add_argument(
    "--wait-sleep-s",
    type=float,
    default=0.002,
    help="Wall-clock sleep used while strict DDS wait is active.",
)
parser.add_argument(
    "--strict-wait-timeout-s",
    type=float,
    default=30.0,
    help=(
        "Timeout for --strict-wait-dds before aborting with diagnostics. "
        "Set <=0 to disable timeout."
    ),
)
parser.add_argument(
    "--done-behavior",
    type=str,
    choices=("reset", "stop"),
    default="reset",
    help=(
        "Behavior when done=True in replay loop. "
        "reset: print reason and reset env (official-like sim2sim). "
        "stop: keep legacy behavior and exit loop."
    ),
)
parser.add_argument(
    "--dds-target-order",
    type=str,
    choices=("auto", "mujoco", "isaaclab"),
    default="auto",
    help=(
        "Expected joint order of DDS target stream from C++ deploy. "
        "auto: use tracker_config dds.target_order (default mujoco)."
    ),
)
parser.add_argument(
    "--joint-diag-topk",
    type=int,
    default=6,
    help="In --diagnose mode, print top-k joints by |target-q_meas| every diag interval.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from lwmg.references.replay_reference import ReplayReferenceGenerator
from lwmg.sonic_io import (
    SonicDdsBridge,
    SonicDdsBridgeConfig,
    SonicZmqManagerControl,
    SonicZmqManagerControlConfig,
)
from lwmg.tasks.direct.lwmg.lwmg_env import LwmgEnv
from lwmg.tasks.direct.lwmg.lwmg_env_cfg import LwmgEnvCfg

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
    return (base_dir / candidate).resolve()


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got type={type(data).__name__}")
    return data


def _build_direct_env(env_cfg: dict[str, Any], bridge_cfg: dict[str, Any], num_envs: int) -> LwmgEnv:
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

    env_block = env_cfg.get("env", {})
    if isinstance(env_block, dict) and "episode_length_s" in env_block:
        cfg.episode_length_s = float(env_block["episode_length_s"])

    term_cfg = env_cfg.get("terminations", {})
    if isinstance(term_cfg, dict):
        hard_cfg = term_cfg.get("hard", {})
        soft_cfg = term_cfg.get("soft", {})

        if isinstance(hard_cfg, dict):
            if "base_height_min_m" in hard_cfg:
                cfg.min_base_height = float(hard_cfg["base_height_min_m"])
            if "trunk_tilt_fail_rad" in hard_cfg:
                cfg.trunk_tilt_fail_rad = float(hard_cfg["trunk_tilt_fail_rad"])
            else:
                fail_pitch = hard_cfg.get("trunk_pitch_fail_rad", None)
                fail_roll = hard_cfg.get("trunk_roll_fail_rad", None)
                fail_candidates = [float(v) for v in (fail_pitch, fail_roll) if v is not None]
                if fail_candidates:
                    cfg.trunk_tilt_fail_rad = min(fail_candidates)

        if isinstance(soft_cfg, dict):
            if "trunk_tilt_warn_rad" in soft_cfg:
                cfg.trunk_tilt_warn_rad = float(soft_cfg["trunk_tilt_warn_rad"])
            else:
                warn_pitch = soft_cfg.get("trunk_pitch_warn_rad", None)
                warn_roll = soft_cfg.get("trunk_roll_warn_rad", None)
                warn_candidates = [float(v) for v in (warn_pitch, warn_roll) if v is not None]
                if warn_candidates:
                    cfg.trunk_tilt_warn_rad = min(warn_candidates)

    # DDS receives absolute q_target from official C++ CreatePolicyCommand.
    cfg.control_mode = "joint_target_absolute"
    cfg.action_scale = 1.0

    action_cfg = bridge_cfg.get("action", {}) if isinstance(bridge_cfg.get("action", {}), dict) else {}
    cfg.clamp_to_joint_limits = bool(action_cfg.get("clamp_to_joint_limits", False))

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


def _tensor_stats(x: torch.Tensor) -> dict[str, float]:
    flat = x.detach().to(dtype=torch.float32).flatten()
    if flat.numel() == 0:
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "abs_mean": 0.0,
            "abs_max": 0.0,
            "abs_p95": 0.0,
        }
    abs_flat = flat.abs()
    return {
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
        "abs_mean": float(abs_flat.mean().item()),
        "abs_max": float(abs_flat.max().item()),
        "abs_p95": float(torch.quantile(abs_flat, 0.95).item()),
    }


def _format_stats(name: str, stats: dict[str, float]) -> str:
    return (
        f"{name}: mean={stats['mean']:.4f} std={stats['std']:.4f} "
        f"min={stats['min']:.4f} max={stats['max']:.4f} "
        f"|abs|mean={stats['abs_mean']:.4f} |abs|max={stats['abs_max']:.4f} "
        f"|abs|p95={stats['abs_p95']:.4f}"
    )


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _init_trace_store() -> dict[str, list[Any]]:
    return {
        "step": [],
        "frame_idx": [],
        "reward": [],
        "done": [],
        "cmd_fresh": [],
        "cmd_stale": [],
        "cmd_age_s": [],
        "q_meas_abs_pre": [],
        "dq_meas_pre": [],
        "mapped_action": [],
        "dds_target_raw": [],
        "dds_target_mujoco": [],
        "clamped_fraction": [],
        "q_ref_abs": [],
        "dq_ref": [],
    }


def _save_trace_npz(trace: dict[str, list[Any]], out_path: Path, *, meta: dict[str, Any]) -> None:
    assert np is not None

    arrays: dict[str, np.ndarray] = {}
    for key, values in trace.items():
        if not values:
            continue
        first = values[0]
        if isinstance(first, np.ndarray):
            try:
                arrays[key] = np.stack(values, axis=0)
            except ValueError:
                arrays[key] = np.array(values, dtype=object)
            continue
        arrays[key] = np.asarray(values)

    for k, v in meta.items():
        arrays[f"meta_{k}"] = np.asarray(v)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **arrays)
    print(f"[replay_motion] trace saved: {out_path}")


def _build_control_input_client(tracker_cfg_all: dict[str, Any]) -> SonicZmqManagerControl | None:
    block = tracker_cfg_all.get("control_input", {})
    if not isinstance(block, dict):
        return None

    mode = str(block.get("mode", "none")).strip().lower()
    if mode in {"", "none", "off", "disabled", "false"}:
        return None
    if mode != "zmq_manager":
        raise ValueError(f"Unsupported control_input.mode={mode}. Use none|zmq_manager")

    zmq_block = block.get("zmq_manager", {})
    if not isinstance(zmq_block, dict):
        raise ValueError("control_input.zmq_manager must be a mapping")

    cfg = SonicZmqManagerControlConfig.from_dict(zmq_block)
    if not cfg.enabled:
        return None

    client = SonicZmqManagerControl(cfg)
    client.start()
    return client


def _check_strict_wait_timeout(
    *,
    timeout_s: float,
    wait_started_at: float,
    wait_loops: int,
    cmd_status: dict[str, float | bool],
    bridge_cfg: SonicDdsBridgeConfig,
    control_input: SonicZmqManagerControl | None,
) -> None:
    if timeout_s <= 0.0:
        return
    elapsed_s = max(0.0, time.monotonic() - wait_started_at)
    if elapsed_s <= timeout_s:
        return

    endpoint = control_input.endpoint if control_input is not None else "disabled"
    raise RuntimeError(
        "Timed out while waiting for first DDS command in --strict-wait-dds mode.\n"
        f"  waited_s={elapsed_s:.3f} timeout_s={timeout_s:.3f} loops={wait_loops}\n"
        f"  cmd_fresh={int(bool(cmd_status.get('fresh', False)))} "
        f"cmd_stale={int(bool(cmd_status.get('stale', True)))} "
        f"cmd_age_s={float(cmd_status.get('age_s', float('inf'))):.3f}\n"
        f"  dds.domain_id={bridge_cfg.domain_id} dds.interface={bridge_cfg.interface or ''} "
        f"control_input_endpoint={endpoint}\n"
        "Quick checks:\n"
        "  1) C++ deploy is running and not exiting early.\n"
        "  2) --motion-data points to motion root directory (e.g. outputs/sonic_refs), not clip_000.\n"
        "  3) No stale process is occupying tcp://127.0.0.1:5556.\n"
        "  4) DDS domain/interface match on both sides."
    )


def _format_done_reason(diagnostics: dict[str, torch.Tensor], env_id: int) -> str:
    base_h = float(diagnostics["base_height"][env_id].item())
    tilt = float(diagnostics["tilt"][env_id].item())
    numeric = bool(diagnostics["numeric_fail"][env_id].item())
    timeout = bool(diagnostics["timeout"][env_id].item())

    reasons: list[str] = []
    if bool(diagnostics["base_height_fail"][env_id].item()):
        reasons.append("base_height_fail")
    if bool(diagnostics["tilt_fail"][env_id].item()):
        reasons.append("tilt_fail")
    if numeric:
        reasons.append("numeric_fail")
    if timeout:
        reasons.append("timeout")

    if not reasons:
        reasons.append("unknown")

    return (
        f"reasons={','.join(reasons)} "
        f"base_h={base_h:.4f} "
        f"tilt={tilt:.4f} "
        f"numeric={int(numeric)} timeout={int(timeout)}"
    )


def _resolve_dds_target_order(dds_block: dict[str, Any], cli_value: str) -> str:
    if cli_value != "auto":
        return cli_value
    configured = str(dds_block.get("target_order", "mujoco")).strip().lower()
    if configured not in {"mujoco", "isaaclab"}:
        raise ValueError(f"Invalid dds.target_order={configured}. Use mujoco|isaaclab.")
    return configured


def _resolve_dds_custom_target_map(dds_block: dict[str, Any], *, num_joints: int) -> list[int] | None:
    raw = dds_block.get("target_to_isaaclab_index_map", None)
    if raw is None:
        return None
    mapping = [int(v) for v in list(raw)]
    if len(mapping) != num_joints:
        raise ValueError(
            f"dds.target_to_isaaclab_index_map len={len(mapping)} does not match num_joints={num_joints}"
        )
    return mapping


def _format_top_joint_abs_error(
    err: torch.Tensor,
    joint_names: list[str],
    *,
    topk: int,
) -> str:
    abs_err = err.detach().abs().to(dtype=torch.float32)
    k = max(1, min(int(topk), abs_err.numel()))
    values, indices = torch.topk(abs_err, k=k, largest=True)
    parts: list[str] = []
    for value, idx in zip(values.tolist(), indices.tolist(), strict=False):
        if 0 <= idx < len(joint_names):
            name = joint_names[idx]
        else:
            name = f"joint_{idx}"
        parts.append(f"{name}[{idx}]={value:.4f}")
    return ", ".join(parts)


def main() -> None:
    env_cfg_all = _load_yaml(args_cli.env_config)
    tracker_cfg_all = _load_yaml(args_cli.tracker_config)

    dds_block = tracker_cfg_all.get("dds", {})
    if not isinstance(dds_block, dict) or not dds_block:
        # compatibility: accept top-level fields directly in tracker config
        dds_block = tracker_cfg_all

    if int(args_cli.num_envs) != 1:
        raise ValueError("DDS replay currently supports --num-envs 1 only (single SONIC C++ deploy process).")

    direct_env = _build_direct_env(env_cfg_all, bridge_cfg=dds_block, num_envs=args_cli.num_envs)
    replay_gen = _build_reference_generator(
        env_cfg=env_cfg_all,
        explicit_clip=args_cli.reference_clip,
        joint_dim=direct_env.num_joints,
        device=str(direct_env.device),
        allow_degenerate_clip=args_cli.allow_degenerate_reference,
    )

    bridge_cfg = SonicDdsBridgeConfig.from_dict(dds_block)
    bridge = SonicDdsBridge(bridge_cfg)
    dds_target_order = _resolve_dds_target_order(dds_block, args_cli.dds_target_order)
    dds_target_custom_map = _resolve_dds_custom_target_map(dds_block, num_joints=direct_env.num_joints)

    control_input = _build_control_input_client(tracker_cfg_all)
    if control_input is None:
        print("[replay_motion] control_input: disabled (no auto start command publisher)")
    else:
        print(
            f"[replay_motion] control_input=zmq_manager endpoint={control_input.endpoint} "
            f"planner_mode={control_input.cfg.planner_mode} auto_start={control_input.cfg.auto_start}"
        )

    if args_cli.stream_reference:
        if replay_gen is None:
            raise ValueError("--stream-reference requires --reference-clip or env.reference.clip_dir.")
        if control_input is None:
            raise ValueError("--stream-reference requires control_input.mode=zmq_manager in tracker config.")
        if control_input.cfg.planner_mode:
            raise ValueError("--stream-reference requires planner_mode=false (STREAMED_MOTION mode).")
        if np is None:
            raise RuntimeError("--stream-reference requires numpy.")

    sim_dt = float(direct_env.cfg.sim.dt)
    decimation = int(direct_env.cfg.decimation)
    control_dt = sim_dt * float(decimation)

    print(
        f"[replay_motion] mode=dds step_mode={args_cli.step_mode} "
        f"sim_dt={sim_dt:.6f}s decimation={decimation} control_dt={control_dt:.6f}s num_envs={direct_env.num_envs} "
        f"dds.domain_id={bridge_cfg.domain_id} dds.interface={bridge_cfg.interface} "
        f"dds.target_order={dds_target_order} custom_map={int(dds_target_custom_map is not None)}"
    )
    print(
        f"[replay_motion] env.control_mode={direct_env.cfg.control_mode} "
        f"clamp_to_joint_limits={bool(getattr(direct_env.cfg, 'clamp_to_joint_limits', True))} "
        f"stream_reference={bool(args_cli.stream_reference)} strict_wait_dds={bool(args_cli.strict_wait_dds)}"
    )

    if replay_gen is None:
        print("[replay_motion] reference clip: disabled (no clip_dir/--reference-clip provided)")
    else:
        q_abs_max = float(torch.max(torch.abs(replay_gen.joint_pos)).item())
        dq_abs_max = float(torch.max(torch.abs(replay_gen.joint_vel)).item())
        root_z_min = float(torch.min(replay_gen.body_pos[:, 2]).item())
        root_z_max = float(torch.max(replay_gen.body_pos[:, 2]).item())
        print(
            f"[replay_motion] reference frames={replay_gen.num_frames} q_abs_max={q_abs_max:.4f} "
            f"dq_abs_max={dq_abs_max:.4f} root_z_range=[{root_z_min:.4f}, {root_z_max:.4f}]"
        )

    reset_out = direct_env.reset()
    _ = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    trace_enabled = args_cli.trace_out is not None
    if trace_enabled and np is None:
        raise RuntimeError("Trace export requires numpy. Please install numpy in the runtime environment.")
    trace_data = _init_trace_store() if trace_enabled else None

    step = 0
    physics_tick = 0
    dds_ever_ready = False
    wait_loops = 0
    strict_wait_started_at: float | None = None
    while simulation_app.is_running() and step < args_cli.max_steps:
        # Use no_grad instead of inference_mode: IsaacLab reset path performs
        # in-place state writes that are incompatible with inference tensors.
        with torch.no_grad():
            if control_input is not None:
                control_input.maybe_send_start_keepalive()

            q_meas_abs_pre, dq_meas_pre = direct_env.get_joint_state_abs_vel(env_id=0)

            # Hold initial pose and freeze reference cursor until C++ DDS command stream is online.
            reference = replay_gen.generate(1) if (replay_gen is not None and dds_ever_ready) else None
            if reference is None:
                q_ref_abs = torch.zeros_like(q_meas_abs_pre).unsqueeze(0)
                dq_ref = torch.zeros_like(dq_meas_pre).unsqueeze(0)
                frame_idx = -1
            else:
                q_ref_abs = reference.joint_pos.to(direct_env.device)
                dq_ref = reference.joint_vel.to(direct_env.device)
                frame_idx = int(reference.frame_idx[0].item()) if reference.frame_idx is not None else -1

            if args_cli.stream_reference and reference is not None and control_input is not None:
                assert np is not None
                frame_index_np = (
                    _to_numpy(reference.frame_idx.to(dtype=torch.int64))
                    if reference.frame_idx is not None
                    else np.asarray([step], dtype=np.int64)
                )
                control_input.send_pose_v1(
                    joint_pos=_to_numpy(reference.joint_pos),
                    joint_vel=_to_numpy(reference.joint_vel),
                    body_quat_w=_to_numpy(reference.root_quat),
                    frame_index=frame_index_np,
                    catch_up=True,
                )

            cmd_status: dict[str, float | bool] = {
                "fresh": False,
                "ever_received": False,
                "stale": True,
                "age_s": float("inf"),
            }
            dds_target_mujoco = np.zeros((direct_env.num_joints,), dtype=np.float32) if np is not None else []
            action = torch.zeros((1, direct_env.num_joints), dtype=torch.float32, device=direct_env.device)

            if args_cli.step_mode == "rl":
                payload = direct_env.build_unitree_lowstate_payload(env_id=0, sim_time_s=float(step) * control_dt)
                bridge.publish_low_state(payload)

                dds_target_mujoco, cmd_status = bridge.pull_target_raw()
                dds_ever_ready = dds_ever_ready or bool(cmd_status["ever_received"])
                if args_cli.strict_wait_dds and not dds_ever_ready:
                    strict_wait_started_at = strict_wait_started_at or time.monotonic()
                    wait_loops += 1
                    _check_strict_wait_timeout(
                        timeout_s=float(args_cli.strict_wait_timeout_s),
                        wait_started_at=strict_wait_started_at,
                        wait_loops=wait_loops,
                        cmd_status=cmd_status,
                        bridge_cfg=bridge_cfg,
                        control_input=control_input,
                    )
                    if wait_loops % max(1, args_cli.log_interval) == 0:
                        print(
                            f"[replay_motion] waiting_dds=true loops={wait_loops} "
                            f"cmd_fresh={int(bool(cmd_status['fresh']))} cmd_stale={int(bool(cmd_status['stale']))} "
                            f"cmd_age_s={float(cmd_status['age_s']):.3f}"
                        )
                    time.sleep(max(0.0, float(args_cli.wait_sleep_s)))
                    continue
                if dds_ever_ready:
                    strict_wait_started_at = None
                    target_isaac = torch.tensor(
                        SonicDdsBridge.target_raw_to_isaaclab(
                            dds_target_mujoco,
                            target_order=dds_target_order,
                            target_to_isaaclab_index_map=dds_target_custom_map,
                        ),
                        dtype=torch.float32,
                        device=direct_env.device,
                    )
                    action = target_isaac.unsqueeze(0)
                else:
                    action = q_meas_abs_pre.unsqueeze(0)

                step_out = direct_env.step(action)
                if isinstance(step_out, tuple) and len(step_out) == 5:
                    _obs, reward, terminated, truncated, _info = step_out
                    done = terminated | truncated
                elif isinstance(step_out, tuple) and len(step_out) == 4:
                    _obs, reward, done, _info = step_out
                else:
                    raise RuntimeError("Unexpected step() return structure from direct environment")
            else:
                if args_cli.strict_wait_dds and not dds_ever_ready:
                    sim_time_s = float(physics_tick) * sim_dt
                    payload = direct_env.build_unitree_lowstate_payload(
                        env_id=0,
                        sim_time_s=sim_time_s,
                        dt_s=sim_dt,
                    )
                    bridge.publish_low_state(payload)

                    dds_target_mujoco, cmd_status = bridge.pull_target_raw()
                    dds_ever_ready = dds_ever_ready or bool(cmd_status["ever_received"])
                    if not dds_ever_ready:
                        strict_wait_started_at = strict_wait_started_at or time.monotonic()
                        wait_loops += 1
                        _check_strict_wait_timeout(
                            timeout_s=float(args_cli.strict_wait_timeout_s),
                            wait_started_at=strict_wait_started_at,
                            wait_loops=wait_loops,
                            cmd_status=cmd_status,
                            bridge_cfg=bridge_cfg,
                            control_input=control_input,
                        )
                        if wait_loops % max(1, args_cli.log_interval) == 0:
                            print(
                                f"[replay_motion] waiting_dds=true loops={wait_loops} "
                                f"cmd_fresh={int(bool(cmd_status['fresh']))} cmd_stale={int(bool(cmd_status['stale']))} "
                                f"cmd_age_s={float(cmd_status['age_s']):.3f}"
                            )
                        time.sleep(max(0.0, float(args_cli.wait_sleep_s)))
                        continue
                    strict_wait_started_at = None

                for _sub in range(decimation):
                    sim_time_s = float(physics_tick) * sim_dt
                    payload = direct_env.build_unitree_lowstate_payload(
                        env_id=0,
                        sim_time_s=sim_time_s,
                        dt_s=sim_dt,
                    )
                    bridge.publish_low_state(payload)

                    dds_target_mujoco, cmd_status = bridge.pull_target_raw()
                    dds_ever_ready = dds_ever_ready or bool(cmd_status["ever_received"])
                    if dds_ever_ready:
                        target_isaac = torch.tensor(
                            SonicDdsBridge.target_raw_to_isaaclab(
                                dds_target_mujoco,
                                target_order=dds_target_order,
                                target_to_isaaclab_index_map=dds_target_custom_map,
                            ),
                            dtype=torch.float32,
                            device=direct_env.device,
                        )
                        action = target_isaac.unsqueeze(0)
                    else:
                        q_hold, _ = direct_env.get_joint_state_abs_vel(env_id=0)
                        action = q_hold.unsqueeze(0)
                    direct_env.step_physics(action)
                    physics_tick += 1

                terminated, truncated = direct_env.get_done_flags()
                done = terminated | truncated

                reward = None
                if args_cli.report_rl_signals:
                    reward, _term_dbg, _trunc_dbg = direct_env.get_rl_signals()

            q_ref_minus_meas = q_ref_abs[0] - q_meas_abs_pre

        reward_value = float("nan") if reward is None else float(reward[0].item())

        if trace_enabled and trace_data is not None:
            trace_data["step"].append(step)
            trace_data["frame_idx"].append(frame_idx)
            trace_data["reward"].append(reward_value)
            trace_data["done"].append(int(done[0].item()))
            trace_data["cmd_fresh"].append(int(bool(cmd_status["fresh"])))
            trace_data["cmd_stale"].append(int(bool(cmd_status["stale"])))
            trace_data["cmd_age_s"].append(float(cmd_status["age_s"]))
            trace_data["q_meas_abs_pre"].append(_to_numpy(q_meas_abs_pre))
            trace_data["dq_meas_pre"].append(_to_numpy(dq_meas_pre))
            trace_data["mapped_action"].append(_to_numpy(action[0]))
            dds_target_raw_np = np.asarray(dds_target_mujoco, dtype=np.float32)
            trace_data["dds_target_raw"].append(dds_target_raw_np)
            trace_data["dds_target_mujoco"].append(dds_target_raw_np)
            trace_data["clamped_fraction"].append(float(direct_env.last_clamped_fraction[0].item()))
            trace_data["q_ref_abs"].append(_to_numpy(q_ref_abs[0]))
            trace_data["dq_ref"].append(_to_numpy(dq_ref[0]))

        if step % max(1, args_cli.log_interval) == 0:
            soft_fail = int(direct_env.soft_failure.sum().item())
            reward_text = "off"
            if reward is not None:
                reward_text = f"{reward.mean().item():.4f}"
            dds_ready_text = "yes" if dds_ever_ready else "no"
            print(
                f"[replay_motion] step={step} reward={reward_text} dds_ready={dds_ready_text} "
                f"hard_fail={int(done.sum().item())} soft_fail={soft_fail} "
                f"cmd_fresh={int(bool(cmd_status['fresh']))} cmd_stale={int(bool(cmd_status['stale']))} "
                f"cmd_age_s={float(cmd_status['age_s']):.3f}"
            )

        if args_cli.diagnose and step % max(1, args_cli.diag_interval) == 0:
            dds_stats = _tensor_stats(torch.tensor(dds_target_mujoco, dtype=torch.float32))
            tgt_stats = _tensor_stats(action[0])
            err_stats = _tensor_stats(q_ref_minus_meas)
            clamp_stats = _tensor_stats(direct_env.last_clamped_fraction)
            target_meas_abs_err = (action[0] - q_meas_abs_pre).abs()
            top_err = _format_top_joint_abs_error(
                target_meas_abs_err,
                list(direct_env.cfg.joint_names),
                topk=args_cli.joint_diag_topk,
            )

            print(f"[replay_diag] {_format_stats(f'dds_target_raw({dds_target_order})', dds_stats)}")
            print(f"[replay_diag] {_format_stats('target_isaac', tgt_stats)}")
            print(f"[replay_diag] {_format_stats('q_ref-q_meas(abs)', err_stats)}")
            print(f"[replay_diag] {_format_stats('clamped_fraction', clamp_stats)}")
            print(f"[replay_diag] target-q_meas top_abs: {top_err}")

            if dds_ever_ready:
                fit_assumed = float(target_meas_abs_err.mean().item())
                if dds_target_custom_map is None and np is not None:
                    alt_order = "isaaclab" if dds_target_order == "mujoco" else "mujoco"
                    alt_target = torch.tensor(
                        SonicDdsBridge.target_raw_to_isaaclab(
                            np.asarray(dds_target_mujoco, dtype=np.float32),
                            target_order=alt_order,
                            target_to_isaaclab_index_map=None,
                        ),
                        dtype=torch.float32,
                        device=direct_env.device,
                    )
                    fit_alt = float(torch.abs(alt_target - q_meas_abs_pre).mean().item())
                    print(
                        f"[replay_diag] mapping_fit: assumed={dds_target_order}:{fit_assumed:.4f} "
                        f"alternative={alt_order}:{fit_alt:.4f} (lower is better)"
                    )

        step += 1

        if args_cli.step_mode == "physics" and bool(done.any().item()):
            diagnostics = direct_env.get_done_diagnostics()
            done_env_ids = torch.nonzero(done, as_tuple=False).flatten().tolist()
            for env_id in done_env_ids:
                reason = _format_done_reason(diagnostics, int(env_id))
                print(f"[replay_motion_done] env={int(env_id)} {reason}")

            if args_cli.done_behavior == "stop":
                print("[replay_motion] done_behavior=stop. stopping replay loop.")
                break

            reset_out = direct_env.reset()
            _ = reset_out[0] if isinstance(reset_out, tuple) else reset_out
            if replay_gen is not None:
                replay_gen.cursor = 0
            print(
                "[replay_motion] done_behavior=reset. env reset to initial state; "
                "continuing replay."
            )

    if trace_enabled and trace_data is not None and args_cli.trace_out is not None:
        meta = {
            "num_envs": direct_env.num_envs,
            "sim_dt": sim_dt,
            "decimation": decimation,
            "control_dt": control_dt,
            "step_mode": args_cli.step_mode,
            "stream_reference": bool(args_cli.stream_reference),
            "report_rl_signals": bool(args_cli.report_rl_signals),
            "strict_wait_dds": bool(args_cli.strict_wait_dds),
            "wait_sleep_s": float(args_cli.wait_sleep_s),
            "strict_wait_timeout_s": float(args_cli.strict_wait_timeout_s),
            "dds_domain_id": bridge_cfg.domain_id,
            "dds_interface": bridge_cfg.interface or "",
            "control_input_mode": "zmq_manager" if control_input is not None else "none",
            "control_input_endpoint": control_input.endpoint if control_input is not None else "",
            "control_input_planner_mode": int(control_input.cfg.planner_mode) if control_input is not None else -1,
            "dds_target_order": dds_target_order,
            "dds_custom_target_map": int(dds_target_custom_map is not None),
            "done_behavior": args_cli.done_behavior,
        }
        _save_trace_npz(trace=trace_data, out_path=args_cli.trace_out, meta=meta)

    if control_input is not None:
        control_input.close()

    direct_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
