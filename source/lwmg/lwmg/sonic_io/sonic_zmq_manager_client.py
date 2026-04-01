from __future__ import annotations

import json
import struct
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class SonicZmqManagerControlConfig:
    """Config for publishing official ZMQManager command messages."""

    enabled: bool = True
    bind_host: str = "127.0.0.1"
    bind_port: int = 5556
    command_topic: str = "command"
    pose_topic: str = "pose"
    auto_start: bool = True
    auto_stop: bool = True
    planner_mode: bool = False
    resend_interval_s: float = 0.50
    startup_delay_s: float = 0.50
    stop_repeat: int = 3
    stop_repeat_interval_s: float = 0.05
    verbose: bool = True

    @classmethod
    def from_dict(cls, data: dict[str, Any] | None) -> "SonicZmqManagerControlConfig":
        raw = data or {}
        return cls(
            enabled=bool(raw.get("enabled", True)),
            bind_host=str(raw.get("bind_host", "127.0.0.1")),
            bind_port=int(raw.get("bind_port", 5556)),
            command_topic=str(raw.get("command_topic", "command")),
            pose_topic=str(raw.get("pose_topic", "pose")),
            auto_start=bool(raw.get("auto_start", True)),
            auto_stop=bool(raw.get("auto_stop", True)),
            planner_mode=bool(raw.get("planner_mode", False)),
            resend_interval_s=float(raw.get("resend_interval_s", 0.50)),
            startup_delay_s=float(raw.get("startup_delay_s", 0.50)),
            stop_repeat=int(raw.get("stop_repeat", 3)),
            stop_repeat_interval_s=float(raw.get("stop_repeat_interval_s", 0.05)),
            verbose=bool(raw.get("verbose", True)),
        )


class SonicZmqManagerControl:
    """Publish official ZMQManager packed messages (command + pose)."""

    HEADER_SIZE = 1280  # Must match ZMQPackedMessageSubscriber::HEADER_SIZE

    def __init__(self, cfg: SonicZmqManagerControlConfig):
        self.cfg = cfg
        self._zmq = None
        self._ctx = None
        self._socket = None
        self._endpoint = f"tcp://{self.cfg.bind_host}:{self.cfg.bind_port}"
        self._last_start_sent_at: float | None = None

    @property
    def endpoint(self) -> str:
        return self._endpoint

    def start(self) -> None:
        if not self.cfg.enabled:
            return

        try:
            import zmq  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pyzmq is required for SONIC zmq_manager auto control mode. Install dependency: pyzmq"
            ) from exc

        self._zmq = zmq
        self._ctx = zmq.Context.instance()
        self._socket = self._ctx.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 3)
        self._socket.setsockopt(zmq.LINGER, 0)
        self._socket.setsockopt(zmq.IMMEDIATE, 1)
        self._socket.bind(self._endpoint)

        if self.cfg.verbose:
            print(
                f"[sonic_zmq_manager] bound={self._endpoint} "
                f"command_topic={self.cfg.command_topic} pose_topic={self.cfg.pose_topic} "
                f"planner_mode={self.cfg.planner_mode} auto_start={self.cfg.auto_start}"
            )

        if self.cfg.startup_delay_s > 0.0:
            time.sleep(self.cfg.startup_delay_s)

        if self.cfg.auto_start:
            self.send_command(start=True, stop=False, planner=self.cfg.planner_mode)

    def _ensure_started(self) -> None:
        if self._socket is None:
            raise RuntimeError("SonicZmqManagerControl is not started. Call start() first.")

    def _pack_header(self, *, version: int, count: int, fields: list[dict[str, Any]]) -> bytes:
        header = {
            "v": int(version),
            "endian": "le",
            "count": int(count),
            "fields": fields,
        }
        header_json = json.dumps(header, separators=(",", ":")).encode("utf-8")
        if len(header_json) > self.HEADER_SIZE:
            raise ValueError(f"ZMQ header too long: {len(header_json)} > {self.HEADER_SIZE}")
        return header_json + (b"\x00" * (self.HEADER_SIZE - len(header_json)))

    def _build_command_message(self, *, start: bool, stop: bool, planner: bool, delta_heading: float | None) -> bytes:
        fields: list[dict[str, Any]] = [
            {"name": "start", "dtype": "u8", "shape": [1]},
            {"name": "stop", "dtype": "u8", "shape": [1]},
            {"name": "planner", "dtype": "u8", "shape": [1]},
        ]
        payload = struct.pack("BBB", 1 if start else 0, 1 if stop else 0, 1 if planner else 0)

        if delta_heading is not None:
            fields.append({"name": "delta_heading", "dtype": "f32", "shape": [1]})
            payload += struct.pack("<f", float(delta_heading))

        header_bytes = self._pack_header(version=1, count=1, fields=fields)
        topic = self.cfg.command_topic.encode("utf-8")
        return topic + header_bytes + payload

    def send_command(self, *, start: bool, stop: bool, planner: bool, delta_heading: float | None = None) -> None:
        self._ensure_started()
        msg = self._build_command_message(start=start, stop=stop, planner=planner, delta_heading=delta_heading)
        assert self._socket is not None
        self._socket.send(msg)

        if start and not stop:
            self._last_start_sent_at = time.monotonic()

        if self.cfg.verbose:
            if delta_heading is None:
                print(f"[sonic_zmq_manager] command start={start} stop={stop} planner={planner}")
            else:
                print(
                    f"[sonic_zmq_manager] command start={start} stop={stop} "
                    f"planner={planner} delta_heading={float(delta_heading):.4f}"
                )

    def send_pose_v1(
        self,
        *,
        joint_pos: np.ndarray,
        joint_vel: np.ndarray,
        body_quat_w: np.ndarray,
        frame_index: np.ndarray,
        catch_up: bool = True,
    ) -> None:
        """Send protocol-v1 pose packet to official ZMQEndpointInterface."""
        self._ensure_started()

        joint_pos = np.asarray(joint_pos, dtype=np.float32)
        joint_vel = np.asarray(joint_vel, dtype=np.float32)
        body_quat_w = np.asarray(body_quat_w, dtype=np.float32)
        frame_index = np.asarray(frame_index, dtype=np.int64)

        if joint_pos.ndim != 2:
            raise ValueError(f"joint_pos must be rank-2 [N, J], got shape={joint_pos.shape}")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"joint_vel shape {joint_vel.shape} must match joint_pos shape {joint_pos.shape}")
        if body_quat_w.ndim != 2 or body_quat_w.shape[1] != 4:
            raise ValueError(f"body_quat_w must be [N, 4], got shape={body_quat_w.shape}")
        if frame_index.ndim != 1:
            raise ValueError(f"frame_index must be rank-1 [N], got shape={frame_index.shape}")

        n_frames = int(joint_pos.shape[0])
        if n_frames <= 0:
            raise ValueError("Cannot stream empty pose packet (N=0).")
        if body_quat_w.shape[0] != n_frames:
            raise ValueError(
                f"body_quat_w frame count mismatch: {body_quat_w.shape[0]} != {n_frames}"
            )
        if frame_index.shape[0] != n_frames:
            raise ValueError(
                f"frame_index length mismatch: {frame_index.shape[0]} != {n_frames}"
            )

        joint_pos = np.ascontiguousarray(joint_pos.astype("<f4", copy=False))
        joint_vel = np.ascontiguousarray(joint_vel.astype("<f4", copy=False))
        body_quat_w = np.ascontiguousarray(body_quat_w.astype("<f4", copy=False))
        frame_index = np.ascontiguousarray(frame_index.astype("<i8", copy=False))

        fields: list[dict[str, Any]] = [
            {"name": "joint_pos", "dtype": "f32", "shape": [n_frames, int(joint_pos.shape[1])]},
            {"name": "joint_vel", "dtype": "f32", "shape": [n_frames, int(joint_vel.shape[1])]},
            {"name": "body_quat_w", "dtype": "f32", "shape": [n_frames, 4]},
            {"name": "frame_index", "dtype": "i64", "shape": [n_frames]},
            {"name": "catch_up", "dtype": "u8", "shape": [1]},
        ]
        header_bytes = self._pack_header(version=1, count=n_frames, fields=fields)

        payload = b"".join(
            (
                joint_pos.tobytes(order="C"),
                joint_vel.tobytes(order="C"),
                body_quat_w.tobytes(order="C"),
                frame_index.tobytes(order="C"),
                struct.pack("B", 1 if catch_up else 0),
            )
        )

        topic = self.cfg.pose_topic.encode("utf-8")
        assert self._socket is not None
        self._socket.send(topic + header_bytes + payload)

        if self.cfg.verbose:
            print(
                "[sonic_zmq_manager] pose v1 "
                f"frames={n_frames} frame_index={int(frame_index[0])}..{int(frame_index[-1])} "
                f"catch_up={bool(catch_up)}"
            )

    def maybe_send_start_keepalive(self) -> None:
        if not self.cfg.enabled or not self.cfg.auto_start:
            return
        if self._socket is None:
            return

        if self.cfg.resend_interval_s <= 0.0:
            return

        now = time.monotonic()
        if self._last_start_sent_at is None or (now - self._last_start_sent_at) >= self.cfg.resend_interval_s:
            self.send_command(start=True, stop=False, planner=self.cfg.planner_mode)

    def close(self) -> None:
        if self._socket is None:
            return

        if self.cfg.enabled and self.cfg.auto_stop:
            repeats = max(1, int(self.cfg.stop_repeat))
            for i in range(repeats):
                try:
                    self.send_command(start=False, stop=True, planner=self.cfg.planner_mode)
                except Exception:
                    break
                if i + 1 < repeats and self.cfg.stop_repeat_interval_s > 0.0:
                    time.sleep(self.cfg.stop_repeat_interval_s)

        try:
            self._socket.close(0)
        finally:
            self._socket = None
