#!/usr/bin/env python3
"""Unified launcher for SONIC replay pipelines (DDS C++ / in-process Python ONNX)."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import yaml


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Tracker config does not exist: {path}")
    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping YAML at {path}, got type={type(data).__name__}")
    return data


def _read_nested(data: dict[str, Any], keys: tuple[str, ...]) -> Any:
    node: Any = data
    for key in keys:
        if not isinstance(node, dict) or key not in node:
            return None
        node = node[key]
    return node


def _resolve_backend(cli_backend: str, tracker_cfg: dict[str, Any]) -> str:
    if cli_backend in {"dds", "python"}:
        return cli_backend

    cfg_backend_candidates = (
        _read_nested(tracker_cfg, ("inference", "backend")),
        _read_nested(tracker_cfg, ("runtime", "inference_backend")),
    )
    for candidate in cfg_backend_candidates:
        if candidate is None:
            continue
        text = str(candidate).strip().lower()
        if text in {"dds", "python"}:
            return text

    dds_block = tracker_cfg.get("dds", {})
    if isinstance(dds_block, dict) and len(dds_block) > 0:
        return "dds"

    paths_block = tracker_cfg.get("paths", {})
    if isinstance(paths_block, dict):
        if "encoder_onnx" in paths_block or "decoder_onnx" in paths_block:
            return "python"

    return "dds"


def _append_flag(cmd: list[str], *, enabled: bool, flag: str) -> None:
    if enabled:
        cmd.append(flag)


def _append_opt(cmd: list[str], *, name: str, value: Any | None) -> None:
    if value is None:
        return
    cmd.extend([name, str(value)])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified launcher: choose SONIC replay backend via CLI or tracker config."
    )
    parser.add_argument("--backend", choices=("auto", "dds", "python"), default="auto")
    parser.add_argument("--env-config", type=Path, required=True)
    parser.add_argument("--tracker-config", type=Path, required=True)
    parser.add_argument("--reference-clip", type=Path, default=None)
    parser.add_argument("--allow-degenerate-reference", action="store_true")
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--diag-interval", type=int, default=50)
    parser.add_argument("--trace-out", type=Path, default=None)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved backend/command and exit.")

    # DDS/C++ chain options
    parser.add_argument("--step-mode", choices=("physics", "rl"), default="physics")
    parser.add_argument("--report-rl-signals", action="store_true")
    parser.add_argument("--stream-reference", action="store_true")
    parser.add_argument("--strict-wait-dds", action="store_true")
    parser.add_argument("--wait-sleep-s", type=float, default=0.002)
    parser.add_argument("--strict-wait-timeout-s", type=float, default=30.0)
    parser.add_argument("--done-behavior", choices=("reset", "stop"), default="reset")
    parser.add_argument("--dds-target-order", choices=("auto", "mujoco", "isaaclab"), default="auto")
    parser.add_argument("--joint-diag-topk", type=int, default=6)

    args, passthrough = parser.parse_known_args()

    tracker_cfg = _load_yaml(args.tracker_config.resolve())
    backend = _resolve_backend(args.backend, tracker_cfg)

    repo_root = Path(__file__).resolve().parents[1]
    target_script = (
        repo_root / "scripts" / "replay_motion.py"
        if backend == "dds"
        else repo_root / "scripts" / "train_sonic_python.py"
    )

    cmd: list[str] = [sys.executable, str(target_script)]
    _append_opt(cmd, name="--env-config", value=args.env_config)
    _append_opt(cmd, name="--tracker-config", value=args.tracker_config)
    _append_opt(cmd, name="--reference-clip", value=args.reference_clip)
    _append_flag(cmd, enabled=args.allow_degenerate_reference, flag="--allow-degenerate-reference")
    _append_opt(cmd, name="--num-envs", value=args.num_envs)
    _append_opt(cmd, name="--max-steps", value=args.max_steps)
    _append_opt(cmd, name="--log-interval", value=args.log_interval)
    _append_flag(cmd, enabled=args.diagnose, flag="--diagnose")
    _append_opt(cmd, name="--diag-interval", value=args.diag_interval)
    _append_opt(cmd, name="--trace-out", value=args.trace_out)
    _append_flag(cmd, enabled=args.headless, flag="--headless")

    if backend == "dds":
        _append_opt(cmd, name="--step-mode", value=args.step_mode)
        _append_flag(cmd, enabled=args.report_rl_signals, flag="--report-rl-signals")
        _append_flag(cmd, enabled=args.stream_reference, flag="--stream-reference")
        _append_flag(cmd, enabled=args.strict_wait_dds, flag="--strict-wait-dds")
        _append_opt(cmd, name="--wait-sleep-s", value=args.wait_sleep_s)
        _append_opt(cmd, name="--strict-wait-timeout-s", value=args.strict_wait_timeout_s)
        _append_opt(cmd, name="--done-behavior", value=args.done_behavior)
        _append_opt(cmd, name="--dds-target-order", value=args.dds_target_order)
        _append_opt(cmd, name="--joint-diag-topk", value=args.joint_diag_topk)
    else:
        # Python in-process chain does not consume DDS-specific arguments.
        ignored = []
        if args.stream_reference:
            ignored.append("--stream-reference")
        if args.strict_wait_dds:
            ignored.append("--strict-wait-dds")
        if args.report_rl_signals:
            ignored.append("--report-rl-signals")
        if ignored:
            print(f"[run_motion_pipeline] backend=python ignored args: {' '.join(ignored)}")

    cmd.extend(passthrough)
    print(f"[run_motion_pipeline] backend={backend} target={target_script.name}")
    print(f"[run_motion_pipeline] exec: {' '.join(cmd)}")
    if args.dry_run:
        return
    os.execv(sys.executable, cmd)


if __name__ == "__main__":
    main()
