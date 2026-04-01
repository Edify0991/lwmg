from __future__ import annotations

import importlib
import json
import os
import select
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from lwmg.tasks.direct.lwmg.g1_robot_cfg import G1_DEFAULT_JOINT_ANGLES, ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB


def _maybe_add_path(path: Path) -> None:
    if path.exists():
        s = str(path)
        if s not in sys.path:
            sys.path.insert(0, s)


def _find_repo_root(start: Path) -> Path | None:
    for parent in [start, *start.parents]:
        if (parent / "thirdparty" / "GR00T-WholeBodyControl").exists():
            return parent
    return None


def _ensure_official_imports() -> tuple[Any, Any]:
    repo_root = _find_repo_root(Path(__file__).resolve())
    if repo_root is not None:
        _maybe_add_path(repo_root / "thirdparty" / "GR00T-WholeBodyControl")
        _maybe_add_path(repo_root / "thirdparty" / "GR00T-WholeBodyControl" / "external_dependencies" / "unitree_sdk2_python")

    try:
        channel_mod = importlib.import_module("unitree_sdk2py.core.channel")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import unitree_sdk2py. "
            "Please install Unitree SDK2 Python bindings or use the thirdparty GR00T repo with dependencies."
        ) from exc

    try:
        bridge_mod = importlib.import_module("gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge")
    except Exception as exc:
        raise RuntimeError(
            "Failed to import official UnitreeSdk2Bridge from gear_sonic. "
            "Expected module: gear_sonic.utils.mujoco_sim.unitree_sdk2py_bridge"
        ) from exc

    return channel_mod.ChannelFactoryInitialize, bridge_mod.UnitreeSdk2Bridge


@dataclass
class SonicDdsBridgeConfig:
    domain_id: int = 0
    interface: str | None = "lo"
    robot_type: str = "g1_29dof"
    num_motors: int = 29
    num_hand_motors: int = 0
    use_sensor: bool = False
    bridge_backend: str = "proxy"  # direct | proxy
    proxy_python: str | None = None
    proxy_script: str | None = None
    proxy_startup_timeout_s: float = 8.0
    proxy_io_timeout_s: float = 1.0
    stale_timeout_s: float = 0.25
    stale_policy: str = "hold_last"  # hold_last | default_pose | zeros
    default_target_isaaclab: list[float] | None = None
    target_order: str = "mujoco"  # mujoco | isaaclab
    target_to_isaaclab_index_map: list[int] | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SonicDdsBridgeConfig":
        raw = data or {}
        default_target = raw.get("default_target_isaaclab", None)
        if default_target is not None:
            default_target = [float(v) for v in list(default_target)]
        target_to_isaaclab = raw.get("target_to_isaaclab_index_map", None)
        if target_to_isaaclab is not None:
            target_to_isaaclab = [int(v) for v in list(target_to_isaaclab)]

        return cls(
            domain_id=int(raw.get("domain_id", 0)),
            interface=None if raw.get("interface", "lo") in {None, "", "null"} else str(raw.get("interface", "lo")),
            robot_type=str(raw.get("robot_type", "g1_29dof")),
            num_motors=int(raw.get("num_motors", 29)),
            num_hand_motors=int(raw.get("num_hand_motors", 0)),
            use_sensor=bool(raw.get("use_sensor", False)),
            bridge_backend=str(raw.get("bridge_backend", "proxy")).strip().lower(),
            proxy_python=None if raw.get("proxy_python", None) in {None, "", "null"} else str(raw.get("proxy_python")),
            proxy_script=None if raw.get("proxy_script", None) in {None, "", "null"} else str(raw.get("proxy_script")),
            proxy_startup_timeout_s=float(raw.get("proxy_startup_timeout_s", 8.0)),
            proxy_io_timeout_s=float(raw.get("proxy_io_timeout_s", 1.0)),
            stale_timeout_s=float(raw.get("stale_timeout_s", 0.25)),
            stale_policy=str(raw.get("stale_policy", "hold_last")).strip().lower(),
            default_target_isaaclab=default_target,
            target_order=str(raw.get("target_order", "mujoco")).strip().lower(),
            target_to_isaaclab_index_map=target_to_isaaclab,
        )


class _UnitreeSdk2ProxyClient:
    """Run official UnitreeSdk2Bridge in a dedicated subprocess and communicate over stdin/stdout JSON lines."""

    def __init__(self, cfg: SonicDdsBridgeConfig):
        self._cfg = cfg
        self._repo_root = _find_repo_root(Path(__file__).resolve())
        self._proc: subprocess.Popen[str] | None = None
        self._io_timeout_s = float(max(0.1, cfg.proxy_io_timeout_s))
        self._startup_timeout_s = float(max(0.5, cfg.proxy_startup_timeout_s))
        self._start()

    def _resolve_path(self, raw: str | None) -> Path | None:
        if raw is None:
            return None
        path = Path(raw).expanduser()
        if path.is_absolute():
            return path
        if self._repo_root is not None:
            return (self._repo_root / path).resolve()
        return path.resolve()

    def _python_has_proxy_deps(self, python_exe: Path) -> bool:
        if not python_exe.exists():
            return False
        probe_env = os.environ.copy()
        probe_env.pop("PYTHONHOME", None)
        probe_env.pop("PYTHONPATH", None)
        probe_env.pop("PYTHONEXECUTABLE", None)
        try:
            result = subprocess.run(
                [str(python_exe), "-c", "import numpy, cyclonedds"],
                env=probe_env,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=4.0,
                check=False,
            )
            return result.returncode == 0
        except Exception:
            return False

    def _resolve_python(self) -> Path:
        candidates: list[Path] = []

        explicit = self._resolve_path(self._cfg.proxy_python)
        if explicit is not None:
            candidates.append(explicit)

        if self._repo_root is not None:
            candidates.append(self._repo_root / "thirdparty" / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python")
            candidates.append(self._repo_root.parent / "GR00T-WholeBodyControl" / ".venv_sim" / "bin" / "python")

        candidates.append(Path(sys.executable))

        seen: set[str] = set()
        for candidate in candidates:
            key = str(candidate)
            if key in seen:
                continue
            seen.add(key)
            if self._python_has_proxy_deps(candidate):
                return candidate

        for candidate in candidates:
            if candidate.exists():
                return candidate

        return Path(sys.executable)

    def _resolve_script(self) -> Path:
        explicit = self._resolve_path(self._cfg.proxy_script)
        if explicit is not None:
            return explicit

        if self._repo_root is None:
            raise RuntimeError(
                "Cannot auto-resolve DDS proxy worker script because repository root was not found. "
                "Please set dds.proxy_script explicitly."
            )
        return (self._repo_root / "source" / "lwmg" / "lwmg" / "sonic_io" / "dds_proxy_worker.py").resolve()

    def _start(self) -> None:
        py_exe = self._resolve_python()
        worker = self._resolve_script()
        if not py_exe.exists():
            raise FileNotFoundError(f"DDS proxy python executable not found: {py_exe}")
        if not worker.exists():
            raise FileNotFoundError(f"DDS proxy worker script not found: {worker}")

        cmd = [
            str(py_exe),
            str(worker),
            "--domain-id",
            str(self._cfg.domain_id),
            "--robot-type",
            str(self._cfg.robot_type),
            "--num-motors",
            str(self._cfg.num_motors),
            "--num-hand-motors",
            str(self._cfg.num_hand_motors),
            "--use-sensor",
            "1" if self._cfg.use_sensor else "0",
        ]
        if self._cfg.interface:
            cmd.extend(["--interface", str(self._cfg.interface)])
        if self._repo_root is not None:
            cmd.extend(["--repo-root", str(self._repo_root)])
        env = os.environ.copy()
        env.pop("PYTHONHOME", None)
        env.pop("PYTHONPATH", None)
        env.pop("PYTHONEXECUTABLE", None)

        self._proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            text=True,
            bufsize=1,
            env=env,
        )
        self._request({"cmd": "ping"}, timeout_s=self._startup_timeout_s)

    def _read_line_json(self, timeout_s: float) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdout is None:
            raise RuntimeError("DDS proxy process is not started.")

        fd = proc.stdout.fileno()
        ready, _, _ = select.select([fd], [], [], timeout_s)
        if not ready:
            raise TimeoutError(f"DDS proxy response timeout after {timeout_s:.3f}s")

        line = proc.stdout.readline()
        if line == "":
            code = proc.poll()
            raise RuntimeError(f"DDS proxy process exited unexpectedly (code={code}).")

        try:
            parsed = json.loads(line)
        except Exception as exc:  # pragma: no cover - defensive
            raise RuntimeError(f"Failed parsing DDS proxy response: {line!r}") from exc
        if not isinstance(parsed, dict):
            raise RuntimeError(f"DDS proxy response must be object, got {type(parsed).__name__}")
        return parsed

    def _request(self, request: dict[str, Any], timeout_s: float | None = None) -> dict[str, Any]:
        proc = self._proc
        if proc is None or proc.stdin is None:
            raise RuntimeError("DDS proxy process is not started.")
        if proc.poll() is not None:
            raise RuntimeError(f"DDS proxy process is not alive (code={proc.returncode}).")

        payload = json.dumps(request, separators=(",", ":"))
        proc.stdin.write(payload + "\n")
        proc.stdin.flush()

        response = self._read_line_json(timeout_s=self._io_timeout_s if timeout_s is None else timeout_s)
        if not bool(response.get("ok", False)):
            err = response.get("error", "unknown")
            trace = response.get("traceback", None)
            if trace:
                raise RuntimeError(f"DDS proxy request failed: {err}\n{trace}")
            raise RuntimeError(f"DDS proxy request failed: {err}")
        return response

    @staticmethod
    def _serialize_payload(payload: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in payload.items():
            if isinstance(value, np.ndarray):
                out[key] = value.astype(np.float32, copy=False).tolist()
            elif isinstance(value, (list, tuple)):
                out[key] = [float(v) for v in value]
            elif isinstance(value, (np.floating, float)):
                out[key] = float(value)
            elif isinstance(value, (np.integer, int)):
                out[key] = int(value)
            else:
                out[key] = value
        return out

    def PublishLowState(self, payload: dict[str, Any]) -> None:
        serialized = self._serialize_payload(payload)
        self._request({"cmd": "publish_low_state", "payload": serialized})

    def GetAction(self) -> tuple[np.ndarray, bool, bool]:
        response = self._request({"cmd": "get_action"})
        body_target = np.asarray(response.get("target", []), dtype=np.float32)
        cmd_received = bool(response.get("cmd_received", False))
        is_new = bool(response.get("is_new", False))
        return body_target, cmd_received, is_new

    def close(self) -> None:
        proc = self._proc
        if proc is None:
            return
        if proc.poll() is None:
            try:
                self._request({"cmd": "shutdown"}, timeout_s=0.5)
            except Exception:
                pass
            try:
                proc.terminate()
                proc.wait(timeout=1.0)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass
        self._proc = None

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


class SonicDdsBridge:
    """IsaacLab-side DDS bridge compatible with official SONIC C++ deploy topics."""

    def __init__(self, cfg: SonicDdsBridgeConfig):
        self.cfg = cfg
        if self.cfg.bridge_backend not in {"direct", "proxy"}:
            raise ValueError(f"Unsupported bridge_backend='{self.cfg.bridge_backend}'. Use direct|proxy.")
        if self.cfg.target_order not in {"mujoco", "isaaclab"}:
            raise ValueError(
                f"Unsupported target_order='{self.cfg.target_order}'. Use mujoco|isaaclab."
            )
        if (
            self.cfg.target_to_isaaclab_index_map is not None
            and len(self.cfg.target_to_isaaclab_index_map) != self.cfg.num_motors
        ):
            raise ValueError(
                f"target_to_isaaclab_index_map len={len(self.cfg.target_to_isaaclab_index_map)} "
                f"does not match num_motors={self.cfg.num_motors}"
            )

        if self.cfg.bridge_backend == "proxy":
            self._bridge = _UnitreeSdk2ProxyClient(self.cfg)
        else:
            channel_factory_init, unitree_bridge_cls = _ensure_official_imports()
            if self.cfg.interface:
                channel_factory_init(self.cfg.domain_id, self.cfg.interface)
            else:
                channel_factory_init(self.cfg.domain_id)

            bridge_cfg = {
                "ROBOT_TYPE": self.cfg.robot_type,
                "NUM_MOTORS": self.cfg.num_motors,
                "NUM_HAND_MOTORS": self.cfg.num_hand_motors,
                "USE_SENSOR": self.cfg.use_sensor,
            }
            self._bridge = unitree_bridge_cls(bridge_cfg)

        default_target_isaac = self.cfg.default_target_isaaclab
        if default_target_isaac is None:
            default_target_isaac = list(G1_DEFAULT_JOINT_ANGLES)
        if len(default_target_isaac) != self.cfg.num_motors:
            raise ValueError(
                f"default_target_isaaclab len={len(default_target_isaac)} does not match num_motors={self.cfg.num_motors}"
            )

        target_isaac = np.asarray(default_target_isaac, dtype=np.float32)
        if self.cfg.target_order == "mujoco":
            self._default_target_raw = target_isaac[np.asarray(ISAACLAB_TO_MUJOCO, dtype=np.int64)]
        else:
            self._default_target_raw = target_isaac.copy()
        self._last_target_raw = self._default_target_raw.copy()

        self._last_rx_time: float | None = None
        self._ever_received = False

        if self.cfg.stale_policy not in {"hold_last", "default_pose", "zeros"}:
            raise ValueError(
                f"Unsupported stale_policy='{self.cfg.stale_policy}'. Use hold_last|default_pose|zeros"
            )

    def publish_low_state(self, low_state_payload: dict[str, Any]) -> None:
        self._bridge.PublishLowState(low_state_payload)

    def _fallback_target(self) -> np.ndarray:
        if self.cfg.stale_policy == "zeros":
            return np.zeros_like(self._default_target_raw)
        if self.cfg.stale_policy == "default_pose":
            return self._default_target_raw.copy()
        return self._last_target_raw.copy()

    def pull_target_raw(self) -> tuple[np.ndarray, dict[str, float | bool]]:
        now = time.monotonic()
        body_target, cmd_received, _is_new = self._bridge.GetAction()

        fresh = False
        if cmd_received:
            arr = np.asarray(body_target[: self.cfg.num_motors], dtype=np.float32)
            if arr.size == self.cfg.num_motors and np.isfinite(arr).all():
                self._last_target_raw = arr.copy()
                self._last_rx_time = now
                self._ever_received = True
                fresh = True

        age_s = float("inf")
        if self._last_rx_time is not None:
            age_s = float(max(0.0, now - self._last_rx_time))

        stale = False
        if self.cfg.stale_timeout_s > 0.0 and self._last_rx_time is not None:
            stale = age_s > self.cfg.stale_timeout_s

        if stale or not self._ever_received:
            target = self._fallback_target()
        else:
            target = self._last_target_raw.copy()

        status: dict[str, float | bool] = {
            "fresh": bool(fresh),
            "ever_received": bool(self._ever_received),
            "stale": bool(stale),
            "age_s": float(age_s),
        }
        return target, status

    def pull_target_mujoco(self) -> tuple[np.ndarray, dict[str, float | bool]]:
        # Backward-compatible alias. Returned array order is configured by cfg.target_order.
        return self.pull_target_raw()

    @staticmethod
    def target_mujoco_to_isaaclab(body_target_mujoco: np.ndarray) -> np.ndarray:
        if body_target_mujoco.ndim != 1:
            raise ValueError(f"Expected rank-1 body_target_mujoco, got shape={body_target_mujoco.shape}")
        if body_target_mujoco.shape[0] != len(MUJOCO_TO_ISAACLAB):
            raise ValueError(
                f"Expected dim={len(MUJOCO_TO_ISAACLAB)} for mujoco target, got {body_target_mujoco.shape[0]}"
            )
        return np.asarray(body_target_mujoco, dtype=np.float32)[np.asarray(MUJOCO_TO_ISAACLAB, dtype=np.int64)]

    @staticmethod
    def target_raw_to_isaaclab(
        body_target_raw: np.ndarray,
        *,
        target_order: str,
        target_to_isaaclab_index_map: list[int] | None = None,
    ) -> np.ndarray:
        arr = np.asarray(body_target_raw, dtype=np.float32)
        if arr.ndim != 1:
            raise ValueError(f"Expected rank-1 body_target_raw, got shape={arr.shape}")

        if target_to_isaaclab_index_map is not None:
            mapping = np.asarray(target_to_isaaclab_index_map, dtype=np.int64)
            if mapping.shape[0] != arr.shape[0]:
                raise ValueError(
                    f"target_to_isaaclab_index_map len={mapping.shape[0]} does not match target dim={arr.shape[0]}"
                )
            return arr[mapping]

        order = str(target_order).strip().lower()
        if order == "isaaclab":
            return arr.copy()
        if order == "mujoco":
            return SonicDdsBridge.target_mujoco_to_isaaclab(arr)
        raise ValueError(f"Unsupported target_order='{target_order}'. Use mujoco|isaaclab.")

    def close(self) -> None:
        close_fn = getattr(self._bridge, "close", None)
        if callable(close_fn):
            close_fn()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass
