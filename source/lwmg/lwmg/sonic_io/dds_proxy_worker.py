from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np


def _write_response(payload: dict[str, Any]) -> None:
    sys.stdout.write(json.dumps(payload, separators=(",", ":")) + "\n")
    sys.stdout.flush()


def _build_bridge(args: argparse.Namespace):
    repo_root = Path(args.repo_root).resolve() if args.repo_root else None
    if repo_root is not None:
        gr00t = repo_root / "thirdparty" / "GR00T-WholeBodyControl"
        unitree = gr00t / "external_dependencies" / "unitree_sdk2_python"
        sys.path.insert(0, str(gr00t))
        sys.path.insert(0, str(unitree))

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize
    from gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge import UnitreeSdk2Bridge

    if args.interface:
        ChannelFactoryInitialize(int(args.domain_id), str(args.interface))
    else:
        ChannelFactoryInitialize(int(args.domain_id))

    bridge_cfg = {
        "ROBOT_TYPE": str(args.robot_type),
        "NUM_MOTORS": int(args.num_motors),
        "NUM_HAND_MOTORS": int(args.num_hand_motors),
        "USE_SENSOR": bool(int(args.use_sensor)),
    }
    return UnitreeSdk2Bridge(bridge_cfg)


def _to_float_array(value: Any) -> np.ndarray:
    return np.asarray(value, dtype=np.float32)


def _decode_low_state(payload: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "time":
            out[key] = float(value)
            continue
        out[key] = _to_float_array(value)
    return out


def _handle_request(bridge: Any, request: dict[str, Any]) -> dict[str, Any]:
    cmd = request.get("cmd", "")

    if cmd == "ping":
        return {"ok": True, "pong": True}

    if cmd == "shutdown":
        return {"ok": True, "shutdown": True}

    if cmd == "publish_low_state":
        payload_raw = request.get("payload", None)
        if not isinstance(payload_raw, dict):
            raise ValueError("publish_low_state expects payload object")
        payload = _decode_low_state(payload_raw)
        bridge.PublishLowState(payload)
        return {"ok": True}

    if cmd == "get_action":
        body_target, cmd_received, is_new = bridge.GetAction()
        target = np.asarray(body_target, dtype=np.float32)
        return {
            "ok": True,
            "target": target.tolist(),
            "cmd_received": bool(cmd_received),
            "is_new": bool(is_new),
        }

    raise ValueError(f"Unsupported command: {cmd!r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Unitree DDS bridge worker process for IsaacLab proxy mode.")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--interface", type=str, default="")
    parser.add_argument("--robot-type", type=str, default="g1_29dof")
    parser.add_argument("--num-motors", type=int, default=29)
    parser.add_argument("--num-hand-motors", type=int, default=0)
    parser.add_argument("--use-sensor", type=int, default=0)
    parser.add_argument("--repo-root", type=str, default="")
    args = parser.parse_args()

    bridge = _build_bridge(args)

    while True:
        line = sys.stdin.readline()
        if line == "":
            break

        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise ValueError("Request must be JSON object")
            response = _handle_request(bridge, request)
        except Exception as exc:
            response = {
                "ok": False,
                "error": str(exc),
                "traceback": traceback.format_exc(),
            }

        _write_response(response)
        if bool(response.get("shutdown", False)):
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
