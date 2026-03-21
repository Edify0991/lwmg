from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import yaml

from lwmg.envs.observations import Observation
from lwmg.sonic_io.sonic_config_parser import parse_observation_config


@dataclass
class SonicObservationBuilder:
    """Builds SONIC observation vectors from Isaac Lab observation dataclasses."""

    mapping_path: Path
    observation_config_path: Path
    debug: bool = False

    def __post_init__(self) -> None:
        mapping_raw = yaml.safe_load(self.mapping_path.read_text()) or {}
        self.mapping: dict[str, str] = mapping_raw.get("mapping", {})
        self.obs_cfg = parse_observation_config(self.observation_config_path)
        requested = int(self.obs_cfg.get("history_steps", 1))
        self.history_steps = 4 if requested >= 4 else 1

    def _ordered_current_fields(self, obs: Observation) -> list[torch.Tensor]:
        source = {
            "q": obs.q,
            "dq": obs.dq,
            "imu_accel": obs.imu_accel,
            "imu_gyro": obs.imu_gyro,
            "prev_action": obs.prev_action,
            "contacts": obs.contacts,
            "tracking_error": obs.tracking_error_summary,
            "tracking_error_summary": obs.tracking_error_summary,
        }
        ordered: list[torch.Tensor] = []
        for key in self.mapping.keys():
            if key not in source:
                raise KeyError(f"Unknown mapping key '{key}' in {self.mapping_path}")
            ordered.append(source[key].flatten())
        return ordered

    def build(self, current: Observation, history: Iterable[Observation] | None = None) -> torch.Tensor:
        history_list = list(history or [])
        required_history = self.history_steps - 1
        if required_history > 0 and len(history_list) < required_history:
            raise ValueError(
                f"history_steps={self.history_steps} requires {required_history} history frames, got {len(history_list)}"
            )

        frames = [current]
        if required_history > 0:
            frames.extend(history_list[:required_history])

        chunks: list[torch.Tensor] = []
        for frame_idx, frame in enumerate(frames):
            frame_chunks = self._ordered_current_fields(frame)
            if not frame_chunks:
                raise ValueError("No mapped features found for SONIC observation construction")
            if self.debug:
                dims = [int(c.numel()) for c in frame_chunks]
                print(f"[SonicObservationBuilder] frame={frame_idx} feature_dims={dims}")
            chunks.extend(frame_chunks)

        out = torch.cat(chunks, dim=0).to(dtype=torch.float32)
        if out.ndim != 1:
            raise RuntimeError(f"Expected flat observation vector, got shape {tuple(out.shape)}")
        if self.debug:
            print(
                "[SonicObservationBuilder] built_obs_dim="
                f"{int(out.numel())} history_steps={self.history_steps} mapping_keys={list(self.mapping.keys())}"
            )
        return out
