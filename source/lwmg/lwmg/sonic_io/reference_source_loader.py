from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

_CANONICAL_JOINT_ORDER_TAGS = {"isaaclab_g1_29", "isaaclab_g1", "g1_29", "sonic_g1_29"}
_CANONICAL_G1_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]


@dataclass
class SonicReferenceData:
    joint_pos: torch.Tensor
    joint_vel: torch.Tensor
    body_pos: torch.Tensor
    body_quat: torch.Tensor
    source_hz: float

    def to_export_dict(self) -> dict[str, torch.Tensor]:
        return {
            "joint_pos": self.joint_pos,
            "joint_vel": self.joint_vel,
            "body_pos": self.body_pos,
            "body_quat": self.body_quat,
        }


def _resolve_path(path_value: Any, base_dir: Path) -> Path:
    if path_value is None:
        raise ValueError("source.path is required")
    path = Path(str(path_value)).expanduser()
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return path


def _normalize_quat_wxyz(quat: torch.Tensor, quat_format: str) -> torch.Tensor:
    fmt = quat_format.lower().strip()
    if fmt in {"xyzw", "qxqyqzqw"}:
        quat = quat[:, [3, 0, 1, 2]]
    elif fmt in {"wxyz", "qwqxqyqz"}:
        pass
    else:
        raise ValueError(f"Unsupported quaternion format '{quat_format}'. Use 'xyzw' or 'wxyz'.")

    norm = torch.linalg.norm(quat, dim=1, keepdim=True).clamp_min(1e-8)
    return quat / norm


def _as_tensor_2d(data: Any, name: str) -> torch.Tensor:
    arr = torch.as_tensor(data, dtype=torch.float32)
    if arr.ndim == 3 and arr.shape[0] == 1:
        arr = arr.squeeze(0)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank-2 [frames, dim], got shape={tuple(arr.shape)}")
    return arr


def _parse_col_selector(selector: Any, name: str) -> list[int | str]:
    if selector is None:
        raise ValueError(f"Missing column selector for '{name}'")

    if isinstance(selector, dict):
        if "cols" in selector:
            raw = selector["cols"]
        elif "start" in selector and "end" in selector:
            start = int(selector["start"])
            end = int(selector["end"])
            if end <= start:
                raise ValueError(f"Invalid range for '{name}': start={start}, end={end}")
            raw = list(range(start, end))
        else:
            raise ValueError(f"Unsupported selector dict for '{name}': {selector}")
    elif isinstance(selector, (list, tuple)):
        raw = list(selector)
    else:
        raw = [selector]

    out: list[int | str] = []
    for item in raw:
        if isinstance(item, (int, np.integer)):
            out.append(int(item))
        else:
            out.append(str(item))
    return out


def _select_cols(df: pd.DataFrame, selector: Any, name: str) -> torch.Tensor:
    cols = _parse_col_selector(selector, name=name)
    try:
        values = df.loc[:, cols].to_numpy(dtype=np.float32)
    except Exception as exc:  # pragma: no cover
        raise ValueError(f"Failed selecting columns for '{name}' with selector={cols}") from exc
    return _as_tensor_2d(values, name)


def _compute_joint_vel(joint_pos: torch.Tensor, hz: float) -> torch.Tensor:
    if hz <= 0:
        raise ValueError(f"Invalid fps/source_hz={hz}")
    if joint_pos.shape[0] <= 1:
        return torch.zeros_like(joint_pos)

    dt = 1.0 / float(hz)
    vel = torch.zeros_like(joint_pos)
    vel[1:] = (joint_pos[1:] - joint_pos[:-1]) / dt
    vel[0] = vel[1]
    return vel


def _resample_linear(signal: torch.Tensor, target_frames: int) -> torch.Tensor:
    if signal.shape[0] == target_frames:
        return signal
    if signal.shape[0] <= 1:
        return signal.repeat(target_frames, 1)

    x = signal.t().unsqueeze(0)
    y = F.interpolate(x, size=target_frames, mode="linear", align_corners=True)
    return y.squeeze(0).t().contiguous()


def _validate_shapes(data: SonicReferenceData, expected_joints: int) -> None:
    n = data.joint_pos.shape[0]
    if data.joint_pos.shape[1] != expected_joints:
        raise ValueError(f"joint_pos expected {expected_joints} joints, got {data.joint_pos.shape[1]}")
    if data.joint_vel.shape != data.joint_pos.shape:
        raise ValueError(
            f"joint_vel shape {tuple(data.joint_vel.shape)} must match joint_pos shape {tuple(data.joint_pos.shape)}"
        )
    if data.body_pos.shape != (n, 3):
        raise ValueError(f"body_pos expected {(n, 3)}, got {tuple(data.body_pos.shape)}")
    if data.body_quat.shape != (n, 4):
        raise ValueError(f"body_quat expected {(n, 4)}, got {tuple(data.body_quat.shape)}")


def _validate_joint_order_metadata(source_cfg: dict[str, Any], expected_joints: int) -> None:
    if not bool(source_cfg.get("strict_joint_order_check", True)):
        return

    joint_order_tag = str(source_cfg.get("joint_order", "isaaclab_g1_29")).lower().strip()
    if joint_order_tag not in _CANONICAL_JOINT_ORDER_TAGS:
        raise ValueError(
            f"source.joint_order={joint_order_tag} is not SONIC G1 canonical order. "
            f"Expected one of {_CANONICAL_JOINT_ORDER_TAGS}"
        )

    names = source_cfg.get("joint_names", None)
    if names is None:
        return

    ordered = [str(name) for name in names]
    if len(ordered) != expected_joints:
        raise ValueError(f"source.joint_names expected length={expected_joints}, got {len(ordered)}")
    if ordered != _CANONICAL_G1_JOINT_NAMES:
        raise ValueError("source.joint_names does not match canonical SONIC G1 joint order")


def _resample_to_target_hz(data: SonicReferenceData, target_hz: float) -> SonicReferenceData:
    src_hz = float(data.source_hz)
    tgt_hz = float(target_hz)
    if src_hz <= 0 or tgt_hz <= 0:
        raise ValueError(f"Invalid source/target hz: {src_hz} -> {tgt_hz}")
    if abs(src_hz - tgt_hz) < 1e-8:
        return data

    n = data.joint_pos.shape[0]
    if n <= 1:
        return SonicReferenceData(
            joint_pos=data.joint_pos.clone(),
            joint_vel=data.joint_vel.clone(),
            body_pos=data.body_pos.clone(),
            body_quat=data.body_quat.clone(),
            source_hz=tgt_hz,
        )

    duration = (n - 1) / src_hz
    target_frames = max(2, int(round(duration * tgt_hz)) + 1)

    joint_pos = _resample_linear(data.joint_pos, target_frames)
    body_pos = _resample_linear(data.body_pos, target_frames)
    body_quat = _normalize_quat_wxyz(_resample_linear(data.body_quat, target_frames), quat_format="wxyz")
    joint_vel = _compute_joint_vel(joint_pos, hz=tgt_hz)

    return SonicReferenceData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos=body_pos,
        body_quat=body_quat,
        source_hz=tgt_hz,
    )


def _load_from_csv(source_cfg: dict[str, Any], base_dir: Path, preset: str) -> SonicReferenceData:
    path = _resolve_path(source_cfg.get("path"), base_dir)
    has_header = bool(source_cfg.get("has_header", False))
    header = 0 if has_header else None

    df = pd.read_csv(path, header=header)
    if not has_header:
        df.columns = list(range(df.shape[1]))

    if preset in {"lafan1_retarget_csv", "amass_retarget_csv"}:
        root_pos_selector = source_cfg.get("root_pos_cols", {"start": 0, "end": 3})
        root_quat_selector = source_cfg.get("root_quat_cols", {"start": 3, "end": 7})
        joint_pos_selector = source_cfg.get("joint_pos_cols", {"start": 7, "end": 36})
    else:
        root_pos_selector = source_cfg.get("root_pos_cols")
        root_quat_selector = source_cfg.get("root_quat_cols")
        joint_pos_selector = source_cfg.get("joint_pos_cols")

    joint_vel_selector = source_cfg.get("joint_vel_cols", None)
    quat_format = str(source_cfg.get("quat_format", "xyzw"))
    source_hz = float(source_cfg.get("fps", 30.0 if preset == "lafan1_retarget_csv" else 60.0 if preset == "amass_retarget_csv" else 50.0))

    body_pos = _select_cols(df, root_pos_selector, "root_pos_cols")
    body_quat = _normalize_quat_wxyz(_select_cols(df, root_quat_selector, "root_quat_cols"), quat_format=quat_format)
    joint_pos = _select_cols(df, joint_pos_selector, "joint_pos_cols")

    if joint_vel_selector is None:
        joint_vel = _compute_joint_vel(joint_pos, hz=source_hz)
    else:
        joint_vel = _select_cols(df, joint_vel_selector, "joint_vel_cols")

    return SonicReferenceData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos=body_pos,
        body_quat=body_quat,
        source_hz=source_hz,
    )


def _load_from_npz(source_cfg: dict[str, Any], base_dir: Path) -> SonicReferenceData:
    path = _resolve_path(source_cfg.get("path"), base_dir)
    payload = np.load(path, allow_pickle=False)

    keys_cfg = source_cfg.get("keys", {})
    joint_pos_key = str(keys_cfg.get("joint_pos", "joint_pos"))
    joint_vel_key = keys_cfg.get("joint_vel", None)
    body_pos_key = str(keys_cfg.get("body_pos", "body_pos"))
    body_quat_key = str(keys_cfg.get("body_quat", "body_quat"))
    quat_format = str(source_cfg.get("quat_format", "wxyz"))
    source_hz = float(source_cfg.get("fps", 50.0))

    if joint_pos_key not in payload:
        raise KeyError(f"NPZ missing key '{joint_pos_key}'")
    if body_pos_key not in payload:
        raise KeyError(f"NPZ missing key '{body_pos_key}'")
    if body_quat_key not in payload:
        raise KeyError(f"NPZ missing key '{body_quat_key}'")

    joint_pos = _as_tensor_2d(payload[joint_pos_key], f"npz:{joint_pos_key}")
    body_pos = _as_tensor_2d(payload[body_pos_key], f"npz:{body_pos_key}")
    body_quat = _normalize_quat_wxyz(_as_tensor_2d(payload[body_quat_key], f"npz:{body_quat_key}"), quat_format=quat_format)

    if joint_vel_key is None:
        joint_vel = _compute_joint_vel(joint_pos, hz=source_hz)
    else:
        joint_vel_key = str(joint_vel_key)
        if joint_vel_key not in payload:
            raise KeyError(f"NPZ missing key '{joint_vel_key}'")
        joint_vel = _as_tensor_2d(payload[joint_vel_key], f"npz:{joint_vel_key}")

    return SonicReferenceData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos=body_pos,
        body_quat=body_quat,
        source_hz=source_hz,
    )


def _load_from_pt(source_cfg: dict[str, Any], base_dir: Path) -> SonicReferenceData:
    path = _resolve_path(source_cfg.get("path"), base_dir)
    payload = torch.load(path, map_location="cpu")
    if not isinstance(payload, dict):
        raise ValueError(f"Expected dict payload in {path}, got {type(payload).__name__}")

    keys_cfg = source_cfg.get("keys", {})
    joint_pos_key = str(keys_cfg.get("joint_pos", "joint_pos"))
    joint_vel_key = keys_cfg.get("joint_vel", None)
    body_pos_key = str(keys_cfg.get("body_pos", "body_pos"))
    body_quat_key = str(keys_cfg.get("body_quat", "body_quat"))
    quat_format = str(source_cfg.get("quat_format", "wxyz"))
    source_hz = float(source_cfg.get("fps", 50.0))

    if joint_pos_key not in payload:
        raise KeyError(f"PT missing key '{joint_pos_key}'")
    if body_pos_key not in payload:
        raise KeyError(f"PT missing key '{body_pos_key}'")
    if body_quat_key not in payload:
        raise KeyError(f"PT missing key '{body_quat_key}'")

    joint_pos = _as_tensor_2d(payload[joint_pos_key], f"pt:{joint_pos_key}")
    body_pos = _as_tensor_2d(payload[body_pos_key], f"pt:{body_pos_key}")
    body_quat = _normalize_quat_wxyz(_as_tensor_2d(payload[body_quat_key], f"pt:{body_quat_key}"), quat_format=quat_format)

    if joint_vel_key is None:
        joint_vel = _compute_joint_vel(joint_pos, hz=source_hz)
    else:
        joint_vel_key = str(joint_vel_key)
        if joint_vel_key not in payload:
            raise KeyError(f"PT missing key '{joint_vel_key}'")
        joint_vel = _as_tensor_2d(payload[joint_vel_key], f"pt:{joint_vel_key}")

    return SonicReferenceData(
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos=body_pos,
        body_quat=body_quat,
        source_hz=source_hz,
    )


def _load_synthetic(source_cfg: dict[str, Any]) -> SonicReferenceData:
    frames = int(source_cfg.get("frames", 100))
    hz = float(source_cfg.get("fps", 50.0))
    return SonicReferenceData(
        joint_pos=torch.zeros(frames, 29, dtype=torch.float32),
        joint_vel=torch.zeros(frames, 29, dtype=torch.float32),
        body_pos=torch.zeros(frames, 3, dtype=torch.float32),
        body_quat=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(frames, 1),
        source_hz=hz,
    )


def load_reference_source(
    source_cfg: dict[str, Any],
    *,
    base_dir: Path,
    expected_joints: int = 29,
    target_hz: float | None = None,
) -> SonicReferenceData:
    source_type = str(source_cfg.get("type", "lafan1_retarget_csv")).lower().strip()
    _validate_joint_order_metadata(source_cfg, expected_joints=expected_joints)

    if source_type in {"lafan1_retarget_csv", "amass_retarget_csv", "generic_csv"}:
        data = _load_from_csv(source_cfg, base_dir=base_dir, preset=source_type)
    elif source_type in {"generated_npz", "npz"}:
        data = _load_from_npz(source_cfg, base_dir=base_dir)
    elif source_type in {"generated_pt", "pt"}:
        data = _load_from_pt(source_cfg, base_dir=base_dir)
    elif source_type in {"synthetic_zero", "zero"}:
        data = _load_synthetic(source_cfg)
    else:
        raise ValueError(
            "Unsupported source.type='{}'. Expected one of: "
            "lafan1_retarget_csv, amass_retarget_csv, generic_csv, generated_npz, generated_pt, synthetic_zero".format(
                source_type
            )
        )

    if target_hz is not None:
        data = _resample_to_target_hz(data, target_hz=float(target_hz))

    _validate_shapes(data, expected_joints=expected_joints)
    return data
