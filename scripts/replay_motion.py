# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause

"""Replay generated references through SONIC frozen tracker or PD fallback."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
from typing import Any

import yaml


def _require_module(module_name: str, install_hint: str) -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(f"Missing module '{module_name}'. {install_hint}")


_require_module("isaaclab.app", "Install Isaac Lab and run this script with the Isaac Lab Python launcher.")

from isaaclab.app import AppLauncher


parser = argparse.ArgumentParser(description="Replay LWMG motion references with tracker-in-the-loop.")
parser.add_argument("--env-config", type=Path, required=True, help="Path to env YAML config.")
parser.add_argument("--tracker-config", type=Path, required=True, help="Path to tracker YAML config.")
parser.add_argument("--use-pd", action="store_true", help="Force PD tracker fallback.")
parser.add_argument("--num-envs", type=int, default=1, help="Number of vectorized environments.")
parser.add_argument("--max-steps", type=int, default=500, help="Max replay steps.")
parser.add_argument("--log-interval", type=int, default=50, help="Print stats every N steps.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Import torch only after app launch as requested.
import torch

from lwmg.envs.humanoid_load_env import make_lwmg_env
from lwmg.references.replay_reference import ReplayReferenceGenerator
from lwmg.trackers.pd_tracker import PDTracker
from lwmg.trackers.sonic_frozen_tracker_adapter import SonicFrozenTrackerAdapter


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file does not exist: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got type={type(data).__name__}")
    return data


def _build_tracker(tracker_cfg: dict[str, Any], use_pd: bool, target_dim: int) -> Any:
    tracker_block = tracker_cfg.get("tracker", {})
    tracker_name = str(tracker_block.get("name", "pd"))

    if use_pd or tracker_name == "pd":
        return PDTracker(kp=float(tracker_block.get("kp", 60.0)), kd=float(tracker_block.get("kd", 4.0)))

    if tracker_name != "frozen_sonic":
        raise ValueError(f"Unsupported tracker '{tracker_name}'. Use 'frozen_sonic' or --use-pd.")

    adapter_cfg_path = Path(tracker_block.get("adapter_cfg", "configs/sonic/sonic_adapter.yaml"))
    obs_mapping_path = Path(tracker_block.get("obs_mapping_cfg", "configs/sonic/sonic_obs_mapping.yaml"))
    adapter_cfg = _load_yaml(adapter_cfg_path).get("sonic", {})

    root_dir = Path(adapter_cfg.get("root_dir", ""))
    if not str(root_dir):
        raise ValueError(
            "SONIC tracker selected but 'sonic.root_dir' is empty in adapter config. "
            "Point it to your downloaded SONIC checkpoint folder."
        )

    return SonicFrozenTrackerAdapter(
        encoder_path=root_dir / adapter_cfg.get("encoder", "model_encoder.onnx"),
        decoder_path=root_dir / adapter_cfg.get("decoder", "model_decoder.onnx"),
        observation_config_path=root_dir / adapter_cfg.get("observation_config", "observation_config.yaml"),
        target_dim=target_dim,
        obs_mapping_path=obs_mapping_path,
        provider=str(adapter_cfg.get("provider", "cpu")),
    )


def _flat_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat([obs["q"], obs["dq"]], dim=-1)


def main() -> None:
    for module_name, hint in [
        ("lwmg.envs.humanoid_load_env", "Ensure the lwmg package is installed in editable mode."),
        ("lwmg.trackers.sonic_frozen_tracker_adapter", "Ensure SONIC adapter modules are available."),
        ("lwmg.trackers.pd_tracker", "Ensure tracker modules are available."),
        ("lwmg.references.replay_reference", "Ensure reference generator modules are available."),
    ]:
        _require_module(module_name, hint)

    env_cfg = _load_yaml(args_cli.env_config)
    tracker_cfg = _load_yaml(args_cli.tracker_config)
    env_cfg.setdefault("env", {})
    env_cfg["env"]["num_envs"] = args_cli.num_envs

    env = make_lwmg_env(env_cfg=env_cfg["env"], device=getattr(args_cli, "device", "cpu"))
    tracker = _build_tracker(tracker_cfg, use_pd=args_cli.use_pd, target_dim=env.num_joints)
    replay_gen = ReplayReferenceGenerator()

    obs = env.reset()
    step = 0
    while simulation_app.is_running() and step < args_cli.max_steps:
        with torch.inference_mode():
            ref = replay_gen.generate(env.num_envs)
            if isinstance(tracker, SonicFrozenTrackerAdapter):
                actions = []
                for env_id in range(env.num_envs):
                    structured = env.single.step(env.q[env_id], env.dq[env_id], env.prev_action[env_id]).observation
                    actions.append(tracker.act_from_structured(structured))
                action = torch.stack(actions, dim=0).to(env.device)
            else:
                action = tracker.act(_flat_obs(obs)) + 0.05 * ref.joint_pos.to(env.device)

            obs, reward, done, info = env.step(action)
            if done.any():
                env.reset_idx(torch.where(done)[0])

        if step % max(1, args_cli.log_interval) == 0:
            print(
                f"[replay_motion] step={step} reward_mean={reward.mean().item():.4f} "
                f"hard_fail={int(done.sum().item())} soft_fail={int(info['soft_failure'].sum().item())}"
            )
        step += 1


if __name__ == "__main__":
    main()
    simulation_app.close()
