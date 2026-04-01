from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from .base_generator import BaseReferenceGenerator
from .reference_types import ReferenceTarget


def _sorted_by_numeric_suffix(columns: list[str]) -> list[str]:
    def _key(name: str) -> tuple[str, int]:
        stem, _, suffix = name.rpartition("_")
        try:
            return stem, int(suffix)
        except ValueError:
            return stem, 0

    return sorted(columns, key=_key)


class ReplayReferenceGenerator(BaseReferenceGenerator):
    def __init__(
        self,
        clip_dir: Path | None = None,
        joint_dim: int = 29,
        device: str = "cpu",
        loop: bool = True,
        allow_degenerate_clip: bool = False,
    ) -> None:
        self.device = torch.device(device)
        self.loop = loop
        self.cursor = 0
        self.joint_dim = joint_dim
        self.allow_degenerate_clip = bool(allow_degenerate_clip)

        if clip_dir is None:
            self.joint_pos = torch.zeros(1, joint_dim, dtype=torch.float32, device=self.device)
            self.joint_vel = torch.zeros(1, joint_dim, dtype=torch.float32, device=self.device)
            self.body_pos = torch.zeros(1, 3, dtype=torch.float32, device=self.device)
            self.body_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=self.device)
            self.clip_dir = None
            return

        self.clip_dir = Path(clip_dir)
        if not self.clip_dir.exists():
            raise FileNotFoundError(f"Reference clip directory not found: {self.clip_dir}")

        self.joint_pos = self._load_tensor(self.clip_dir / "joint_pos.csv", "joint_")
        self.joint_vel = self._load_tensor(self.clip_dir / "joint_vel.csv", "joint_vel_")
        self.body_pos = self._load_tensor(self.clip_dir / "body_pos.csv", "body_pos_")
        self.body_quat = self._load_tensor(self.clip_dir / "body_quat.csv", "body_quat_")

        if self.joint_pos.shape[1] != joint_dim:
            raise ValueError(f"Expected joint_dim={joint_dim}, got {self.joint_pos.shape[1]} from {self.clip_dir}")
        if self.joint_vel.shape != self.joint_pos.shape:
            raise ValueError(
                f"joint_vel shape {tuple(self.joint_vel.shape)} must match joint_pos shape {tuple(self.joint_pos.shape)}"
            )
        if self.body_pos.shape[1] != 3:
            raise ValueError(f"body_pos expected dim=3, got {self.body_pos.shape[1]}")
        if self.body_quat.shape[1] != 4:
            raise ValueError(f"body_quat expected dim=4, got {self.body_quat.shape[1]}")

        if not self.allow_degenerate_clip:
            self._validate_non_degenerate_clip()

    @property
    def num_frames(self) -> int:
        return int(self.joint_pos.shape[0])

    def _load_tensor(self, path: Path, column_prefix: str) -> torch.Tensor:
        if not path.exists():
            raise FileNotFoundError(f"Missing reference file: {path}")
        df = pd.read_csv(path)
        selected = [c for c in df.columns if c.startswith(column_prefix)]
        if not selected:
            selected = list(df.columns)
        selected = _sorted_by_numeric_suffix(selected)
        values = torch.tensor(df[selected].to_numpy(), dtype=torch.float32, device=self.device)
        if values.ndim != 2:
            raise ValueError(f"Expected rank-2 data in {path}, got shape={tuple(values.shape)}")
        return values

    def _validate_non_degenerate_clip(self) -> None:
        q_abs_max = float(torch.max(torch.abs(self.joint_pos)).item())
        dq_abs_max = float(torch.max(torch.abs(self.joint_vel)).item())
        root_z_abs_max = float(torch.max(torch.abs(self.body_pos[:, 2])).item())

        if q_abs_max < 1e-6 and dq_abs_max < 1e-6 and root_z_abs_max < 1e-6:
            clip_dir = str(self.clip_dir) if self.clip_dir is not None else "<none>"
            raise ValueError(
                "Reference clip appears degenerate (joint_pos/joint_vel/root_z are near zero). "
                "This usually means placeholder export data, not a real retargeted motion. "
                "Please export from a real dataset or pass allow_degenerate_clip=True intentionally. "
                f"clip_dir={clip_dir} q_abs_max={q_abs_max:.3e} dq_abs_max={dq_abs_max:.3e} "
                f"root_z_abs_max={root_z_abs_max:.3e}"
            )

    def _window_indices(self, num_frames: int, step: int, start_idx: int | None = None) -> torch.Tensor:
        idx0 = self.cursor if start_idx is None else int(start_idx)
        indices = []
        for i in range(num_frames):
            idx = idx0 + i * step
            if idx >= self.num_frames:
                idx = self.num_frames - 1
            indices.append(idx)
        return torch.tensor(indices, dtype=torch.long, device=self.device)

    def window_joint_pos(self, num_frames: int, step: int, start_idx: int | None = None) -> torch.Tensor:
        return self.joint_pos[self._window_indices(num_frames=num_frames, step=step, start_idx=start_idx)]

    def window_joint_vel(self, num_frames: int, step: int, start_idx: int | None = None) -> torch.Tensor:
        return self.joint_vel[self._window_indices(num_frames=num_frames, step=step, start_idx=start_idx)]

    def window_root_pos(self, num_frames: int, step: int, start_idx: int | None = None) -> torch.Tensor:
        return self.body_pos[self._window_indices(num_frames=num_frames, step=step, start_idx=start_idx)]

    def window_root_quat(self, num_frames: int, step: int, start_idx: int | None = None) -> torch.Tensor:
        return self.body_quat[self._window_indices(num_frames=num_frames, step=step, start_idx=start_idx)]

    def _current_target(self, batch_size: int) -> ReferenceTarget:
        idx = min(self.cursor, self.num_frames - 1)
        q = self.joint_pos[idx].unsqueeze(0).repeat(batch_size, 1)
        dq = self.joint_vel[idx].unsqueeze(0).repeat(batch_size, 1)
        root_pos = self.body_pos[idx].unsqueeze(0).repeat(batch_size, 1)
        root_quat = self.body_quat[idx].unsqueeze(0).repeat(batch_size, 1)
        frame_idx = torch.full((batch_size,), idx, dtype=torch.long, device=self.device)
        return ReferenceTarget(
            joint_pos=q,
            joint_vel=dq,
            root_pos=root_pos,
            root_quat=root_quat,
            frame_idx=frame_idx,
        )

    def generate(self, batch_size: int) -> ReferenceTarget:
        out = self._current_target(batch_size)
        if self.num_frames <= 1:
            return out

        if self.loop:
            self.cursor = (self.cursor + 1) % self.num_frames
        else:
            self.cursor = min(self.cursor + 1, self.num_frames - 1)
        return out
