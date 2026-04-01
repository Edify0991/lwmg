# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Verify Python SONIC ONNX inference against official C++ DDS inference on identical sim states."""

from __future__ import annotations

import argparse
import importlib.util
import math
import os
import re
import time
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None
    from omegaconf import OmegaConf

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
    description="A/B verify Python SONIC ONNX vs official C++ DDS action targets on the same IsaacLab state."
)
parser.add_argument("--env-config", type=Path, required=True, help="Path to env YAML config.")
parser.add_argument(
    "--dds-config",
    type=Path,
    default=Path("configs/tracker/sonic_dds_bridge.yaml"),
    help="Path to DDS bridge YAML config (for C++ communication).",
)
parser.add_argument(
    "--python-tracker-config",
    type=Path,
    default=Path("configs/tracker/frozen_sonic_tracker.yaml"),
    help="Path to Python frozen SONIC tracker config.",
)
parser.add_argument(
    "--reference-clip",
    type=Path,
    required=True,
    help="Reference clip directory containing joint_pos.csv/joint_vel.csv/body_pos.csv/body_quat.csv.",
)
parser.add_argument("--max-steps", type=int, default=1000, help="Max verification control steps.")
parser.add_argument("--log-interval", type=int, default=50, help="Print stats every N steps.")
parser.add_argument(
    "--strict-wait-dds",
    action="store_true",
    help="Do not advance sim before first DDS command arrives.",
)
parser.add_argument(
    "--wait-sleep-s",
    type=float,
    default=0.002,
    help="Sleep interval while waiting for first DDS command.",
)
parser.add_argument(
    "--stream-reference",
    action="store_true",
    help="Publish reference frames to C++ ZMQ manager (must be enabled for strict A/B equivalence).",
)
parser.add_argument(
    "--rms-threshold",
    type=float,
    default=1.0e-2,
    help="PASS threshold for p95 per-step RMS error in q_target (rad).",
)
parser.add_argument(
    "--max-threshold",
    type=float,
    default=5.0e-2,
    help="PASS threshold for global max absolute q_target error (rad).",
)
parser.add_argument(
    "--min-compared-steps",
    type=int,
    default=100,
    help="Minimum compared steps required to declare PASS.",
)
parser.add_argument("--trace-out", type=Path, default=None, help="Optional NPZ path for detailed A/B trace.")
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
    if yaml is not None:
        data = yaml.safe_load(path.read_text()) or {}
    else:
        data = OmegaConf.to_container(OmegaConf.load(str(path)), resolve=True) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got type={type(data).__name__}")
    return data


def _build_direct_env(env_cfg: dict[str, Any]) -> LwmgEnv:
    cfg = LwmgEnvCfg()

    sim_cfg = env_cfg.get("sim", {})
    if "dt" in sim_cfg:
        cfg.sim.dt = float(sim_cfg["dt"])
    if "control_decimation" in sim_cfg:
        cfg.decimation = int(sim_cfg["control_decimation"])
    cfg.sim.render_interval = cfg.decimation

    scene_cfg = env_cfg.get("scene", {})
    cfg.scene.num_envs = 1
    if "env_spacing" in scene_cfg:
        cfg.scene.env_spacing = float(scene_cfg["env_spacing"])

    robot_cfg = env_cfg.get("robot", {})
    if "spawn_height" in robot_cfg:
        cfg.robot_cfg.init_state.pos = (0.0, 0.0, float(robot_cfg["spawn_height"]))
    if "self_collisions" in robot_cfg:
        cfg.robot_cfg.spawn.articulation_props.enabled_self_collisions = bool(robot_cfg["self_collisions"])
    if "usd_path" in robot_cfg:
        cfg.robot_cfg.spawn.usd_path = str(_resolve_path(robot_cfg["usd_path"], Path.cwd()))

    cfg.control_mode = "joint_target_absolute"
    cfg.action_scale = 1.0
    cfg.clamp_to_joint_limits = False

    return LwmgEnv(cfg)


def _build_control_input_client(cfg_all: dict[str, Any]) -> SonicZmqManagerControl | None:
    block = cfg_all.get("control_input", {})
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


def _to_numpy(x: torch.Tensor) -> np.ndarray:
    return x.detach().cpu().numpy()


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    arr = sorted(values)
    if len(arr) == 1:
        return arr[0]
    pos = (len(arr) - 1) * q
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return arr[lo]
    alpha = pos - lo
    return arr[lo] * (1.0 - alpha) + arr[hi] * alpha


def main() -> None:
    env_cfg_all = _load_yaml(args_cli.env_config)
    dds_cfg_all = _load_yaml(args_cli.dds_config)

    dds_block = dds_cfg_all.get("dds", {})
    if not isinstance(dds_block, dict) or not dds_block:
        dds_block = dds_cfg_all

    direct_env = _build_direct_env(env_cfg_all)
    replay_gen = ReplayReferenceGenerator(
        clip_dir=args_cli.reference_clip,
        joint_dim=direct_env.num_joints,
        device=str(direct_env.device),
        loop=True,
        allow_degenerate_clip=False,
    )
    runtime = SonicPythonRuntime.from_tracker_config(
        args_cli.python_tracker_config,
        target_dim=direct_env.num_joints,
        fallback_dir=Path.cwd(),
    )

    bridge_cfg = SonicDdsBridgeConfig.from_dict(dds_block)
    bridge = SonicDdsBridge(bridge_cfg)

    control_input = _build_control_input_client(dds_cfg_all)

    if args_cli.stream_reference:
        if control_input is None:
            raise ValueError("--stream-reference requires control_input.mode=zmq_manager in DDS config")
        if control_input.cfg.planner_mode:
            raise ValueError("--stream-reference requires planner_mode=false")
        if np is None:
            raise RuntimeError("--stream-reference requires numpy")

    if not args_cli.stream_reference:
        print(
            "[verify_sonic_alignment] WARNING: --stream-reference is OFF. "
            "C++ and Python may consume different references, A/B may be invalid."
        )

    print(
        f"[verify_sonic_alignment] sim_dt={float(direct_env.cfg.sim.dt):.6f}s decimation={int(direct_env.cfg.decimation)} "
        f"control_dt={float(direct_env.cfg.sim.dt)*float(direct_env.cfg.decimation):.6f}s"
    )

    reset_out = direct_env.reset()
    _ = reset_out[0] if isinstance(reset_out, tuple) else reset_out

    runtime.reset_all(1)
    py_prev_raw = torch.zeros((1, direct_env.num_joints), dtype=torch.float32, device=direct_env.device)

    compared_steps = 0
    rms_list: list[float] = []
    max_list: list[float] = []
    mean_abs_list: list[float] = []

    trace_enabled = args_cli.trace_out is not None
    trace: dict[str, list[Any]] = {
        "step": [],
        "frame_idx": [],
        "cmd_fresh": [],
        "cmd_stale": [],
        "cmd_age_s": [],
        "q_target_cpp": [],
        "q_target_py": [],
        "diff": [],
        "rms": [],
        "max_abs": [],
        "mean_abs": [],
    }

    dds_ready = False
    step = 0
    wait_loops = 0

    while simulation_app.is_running() and step < args_cli.max_steps:
        with torch.inference_mode():
            if control_input is not None:
                control_input.maybe_send_start_keepalive()

            reference = replay_gen.generate(1) if dds_ready else None
            frame_idx = -1
            if reference is not None and reference.frame_idx is not None and reference.frame_idx.numel() > 0:
                frame_idx = int(reference.frame_idx.flatten()[0].item())

            if args_cli.stream_reference and reference is not None and control_input is not None:
                control_input.send_pose_v1(
                    joint_pos=_to_numpy(reference.joint_pos),
                    joint_vel=_to_numpy(reference.joint_vel),
                    body_quat_w=_to_numpy(reference.root_quat),
                    frame_index=_to_numpy(reference.frame_idx.to(dtype=torch.int64)),
                    catch_up=True,
                )

            direct_env.set_tracker_prev_action(py_prev_raw)
            obs = direct_env.get_tracker_observation(0)

            py_raw, py_target_abs, _debug = runtime.infer_one(
                current=obs,
                reference=reference,
                reference_generator=replay_gen,
                env_id=0,
                return_debug=False,
            )
            py_prev_raw = py_raw.unsqueeze(0)

            payload = direct_env.build_unitree_lowstate_payload(env_id=0, sim_time_s=float(step) * float(direct_env.cfg.sim.dt) * float(direct_env.cfg.decimation))
            bridge.publish_low_state(payload)

            dds_target_mujoco, cmd_status = bridge.pull_target_mujoco()
            dds_ready = dds_ready or bool(cmd_status["ever_received"])

            if args_cli.strict_wait_dds and not dds_ready:
                wait_loops += 1
                if wait_loops % max(1, args_cli.log_interval) == 0:
                    print(
                        f"[verify_sonic_alignment] waiting_dds=true loops={wait_loops} "
                        f"cmd_fresh={int(bool(cmd_status['fresh']))} cmd_stale={int(bool(cmd_status['stale']))}"
                    )
                time.sleep(max(0.0, float(args_cli.wait_sleep_s)))
                continue

            cpp_target_abs = torch.tensor(
                SonicDdsBridge.target_mujoco_to_isaaclab(dds_target_mujoco),
                dtype=torch.float32,
                device=direct_env.device,
            )

            diff = py_target_abs - cpp_target_abs
            rms = float(torch.sqrt(torch.mean(diff * diff)).item())
            max_abs = float(torch.max(torch.abs(diff)).item())
            mean_abs = float(torch.mean(torch.abs(diff)).item())

            # Compare only on fresh C++ command ticks for strict equivalence.
            if bool(cmd_status["fresh"]):
                compared_steps += 1
                rms_list.append(rms)
                max_list.append(max_abs)
                mean_abs_list.append(mean_abs)

                if trace_enabled:
                    trace["step"].append(step)
                    trace["frame_idx"].append(frame_idx)
                    trace["cmd_fresh"].append(int(bool(cmd_status["fresh"])))
                    trace["cmd_stale"].append(int(bool(cmd_status["stale"])))
                    trace["cmd_age_s"].append(float(cmd_status["age_s"]))
                    trace["q_target_cpp"].append(_to_numpy(cpp_target_abs))
                    trace["q_target_py"].append(_to_numpy(py_target_abs))
                    trace["diff"].append(_to_numpy(diff))
                    trace["rms"].append(rms)
                    trace["max_abs"].append(max_abs)
                    trace["mean_abs"].append(mean_abs)

            step_out = direct_env.step(cpp_target_abs.unsqueeze(0))
            if isinstance(step_out, tuple) and len(step_out) == 5:
                _obs, _reward, terminated, truncated, _info = step_out
                done = terminated | truncated
            elif isinstance(step_out, tuple) and len(step_out) == 4:
                _obs, _reward, done, _info = step_out
            else:
                raise RuntimeError("Unexpected step() return structure")

            if bool(done.any().item()):
                direct_env.reset()
                runtime.reset_all(1)
                py_prev_raw.zero_()

        if step % max(1, args_cli.log_interval) == 0:
            p95_rms = _percentile(rms_list, 0.95) if rms_list else float("nan")
            max_abs_all = max(max_list) if max_list else float("nan")
            print(
                f"[verify_sonic_alignment] step={step} compared={compared_steps} "
                f"p95_rms={p95_rms:.6f} max_abs={max_abs_all:.6f} "
                f"dds_ready={'yes' if dds_ready else 'no'}"
            )

        step += 1

    p50_rms = _percentile(rms_list, 0.50) if rms_list else float("nan")
    p95_rms = _percentile(rms_list, 0.95) if rms_list else float("nan")
    p99_rms = _percentile(rms_list, 0.99) if rms_list else float("nan")
    max_abs_all = max(max_list) if max_list else float("nan")
    mean_abs_all = float(sum(mean_abs_list) / len(mean_abs_list)) if mean_abs_list else float("nan")

    pass_flag = (
        compared_steps >= int(args_cli.min_compared_steps)
        and p95_rms <= float(args_cli.rms_threshold)
        and max_abs_all <= float(args_cli.max_threshold)
    )

    print("[verify_sonic_alignment] ===== SUMMARY =====")
    print(f"compared_steps={compared_steps}")
    print(f"rms_p50={p50_rms:.8f} rms_p95={p95_rms:.8f} rms_p99={p99_rms:.8f}")
    print(f"mean_abs={mean_abs_all:.8f} max_abs={max_abs_all:.8f}")
    print(
        f"thresholds: min_steps={int(args_cli.min_compared_steps)} "
        f"rms_p95<={float(args_cli.rms_threshold):.8f} max_abs<={float(args_cli.max_threshold):.8f}"
    )
    print(f"RESULT={'PASS' if pass_flag else 'FAIL'}")

    if trace_enabled and args_cli.trace_out is not None:
        if np is None:
            raise RuntimeError("Trace export requires numpy")

        arrays: dict[str, np.ndarray] = {}
        for key, values in trace.items():
            if not values:
                continue
            first = values[0]
            if isinstance(first, np.ndarray):
                arrays[key] = np.stack(values, axis=0)
            else:
                arrays[key] = np.asarray(values)

        arrays["meta_compared_steps"] = np.asarray(compared_steps)
        arrays["meta_rms_p50"] = np.asarray(p50_rms)
        arrays["meta_rms_p95"] = np.asarray(p95_rms)
        arrays["meta_rms_p99"] = np.asarray(p99_rms)
        arrays["meta_mean_abs"] = np.asarray(mean_abs_all)
        arrays["meta_max_abs"] = np.asarray(max_abs_all)
        arrays["meta_result_pass"] = np.asarray(int(pass_flag))

        args_cli.trace_out.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args_cli.trace_out, **arrays)
        print(f"[verify_sonic_alignment] trace saved: {args_cli.trace_out}")

    if control_input is not None:
        control_input.close()
    direct_env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
