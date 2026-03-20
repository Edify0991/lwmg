from __future__ import annotations

import csv
from pathlib import Path
from typing import Dict

import torch

from .sonic_motion_format import SonicMotionFormat


class SonicReferenceExporter:
    """Exports references to SONIC-compatible motion folder format."""

    def __init__(self, fmt: SonicMotionFormat | None = None, expected_joints: int = 29) -> None:
        self.fmt = fmt or SonicMotionFormat(frequency_hz=50, joint_order="isaaclab_g1_29")
        self.expected_joints = expected_joints

    def _to_2d(self, tensor: torch.Tensor, name: str) -> torch.Tensor:
        if tensor.ndim != 2:
            raise ValueError(f"{name} must be rank-2 [frames, dim], got shape={tuple(tensor.shape)}")
        return tensor.detach().cpu().to(dtype=torch.float32)

    def _write_csv(self, path: Path, data: torch.Tensor, prefix: str) -> None:
        headers = [f"{prefix}_{i}" for i in range(data.shape[1])]
        with path.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(headers)
            writer.writerows(data.tolist())

    def export(self, out_dir: Path, data: Dict[str, torch.Tensor], clip_name: str = "clip_000") -> Path:
        clip_dir = out_dir / clip_name
        clip_dir.mkdir(parents=True, exist_ok=True)

        joint_pos = self._to_2d(data["joint_pos"], "joint_pos")
        joint_vel = self._to_2d(data["joint_vel"], "joint_vel")
        body_pos = self._to_2d(data["body_pos"], "body_pos")
        body_quat = self._to_2d(data["body_quat"], "body_quat")

        n_frames = joint_pos.shape[0]
        if joint_pos.shape[1] != self.expected_joints:
            raise ValueError(f"joint_pos expected {self.expected_joints} joints, got {joint_pos.shape[1]}")
        if joint_vel.shape != joint_pos.shape:
            raise ValueError(f"joint_vel shape {tuple(joint_vel.shape)} must match joint_pos {tuple(joint_pos.shape)}")
        if body_pos.shape != (n_frames, 3):
            raise ValueError(f"body_pos expected shape {(n_frames, 3)}, got {tuple(body_pos.shape)}")
        if body_quat.shape != (n_frames, 4):
            raise ValueError(f"body_quat expected shape {(n_frames, 4)}, got {tuple(body_quat.shape)}")

        self._write_csv(clip_dir / "joint_pos.csv", joint_pos, "joint")
        self._write_csv(clip_dir / "joint_vel.csv", joint_vel, "joint_vel")
        self._write_csv(clip_dir / "body_pos.csv", body_pos, "body_pos")
        self._write_csv(clip_dir / "body_quat.csv", body_quat, "body_quat")

        metadata = "\n".join(
            [
                "format=sonic_motion_reference",
                f"frequency_hz={self.fmt.frequency_hz}",
                f"joint_order={self.fmt.joint_order}",
                f"num_joints={self.expected_joints}",
                f"num_frames={n_frames}",
            ]
        )
        (clip_dir / "metadata.txt").write_text(metadata + "\n")
        return clip_dir
