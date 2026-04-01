from __future__ import annotations

from dataclasses import dataclass
import os
import re
from pathlib import Path
from typing import Any, Sequence

import torch

try:
    import yaml
except ImportError:
    yaml = None
    from omegaconf import OmegaConf

from lwmg.envs.observations import Observation
from lwmg.references.reference_types import ReferenceTarget
from lwmg.tasks.direct.lwmg.g1_robot_cfg import G1_DEFAULT_JOINT_ANGLES, G1_SONIC_ACTION_SCALE, MUJOCO_TO_ISAACLAB

from .sonic_action_mapper import map_sonic_output_to_targets
from .sonic_frozen_tracker_adapter import SonicFrozenTrackerAdapter


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


def _resolve_path(path_value: Any, cfg_dir: Path, fallback_dir: Path) -> Path:
    raw = _resolve_oc_env(path_value)
    candidate = Path(str(raw)).expanduser()
    if candidate.is_absolute():
        return candidate

    by_cfg = (cfg_dir / candidate).resolve()
    if by_cfg.exists():
        return by_cfg

    return (fallback_dir / candidate).resolve()


@dataclass(frozen=True)
class SonicPythonRuntimeConfig:
    encoder_onnx: Path
    decoder_onnx: Path
    observation_config: Path
    obs_mapping_cfg: Path
    provider: str
    target_dim: int
    batch_mode: bool
    joint_order: str
    clamp_raw_action: bool
    action_clip: float
    action_scale: list[float]
    default_angles: list[float]

    @classmethod
    def from_tracker_config(
        cls,
        tracker_cfg_path: Path,
        *,
        target_dim: int,
        fallback_dir: Path | None = None,
    ) -> "SonicPythonRuntimeConfig":
        cfg_path = tracker_cfg_path.resolve()
        cfg_dir = cfg_path.parent
        project_dir = Path.cwd() if fallback_dir is None else fallback_dir.resolve()

        if yaml is not None:
            raw = yaml.safe_load(cfg_path.read_text()) or {}
        else:
            raw = OmegaConf.to_container(OmegaConf.load(str(cfg_path)), resolve=True) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Tracker config must be a mapping: {cfg_path}")

        paths = raw.get("paths", {}) if isinstance(raw.get("paths", {}), dict) else {}
        adapter = raw.get("adapter", {}) if isinstance(raw.get("adapter", {}), dict) else {}
        tracker = raw.get("tracker", {}) if isinstance(raw.get("tracker", {}), dict) else {}
        action_mapping = raw.get("action_mapping", {}) if isinstance(raw.get("action_mapping", {}), dict) else {}

        encoder_onnx = _resolve_path(paths.get("encoder_onnx"), cfg_dir, project_dir)
        decoder_onnx = _resolve_path(paths.get("decoder_onnx"), cfg_dir, project_dir)
        observation_config = _resolve_path(paths.get("sonic_obs_config"), cfg_dir, project_dir)
        obs_mapping_cfg = _resolve_path(adapter.get("obs_mapping_cfg", "configs/sonic/sonic_obs_mapping.yaml"), cfg_dir, project_dir)

        if not encoder_onnx.exists():
            raise FileNotFoundError(f"SONIC encoder ONNX not found: {encoder_onnx}")
        if not decoder_onnx.exists():
            raise FileNotFoundError(f"SONIC decoder ONNX not found: {decoder_onnx}")
        if not observation_config.exists():
            raise FileNotFoundError(f"SONIC observation config not found: {observation_config}")
        if not obs_mapping_cfg.exists():
            raise FileNotFoundError(f"SONIC observation mapping config not found: {obs_mapping_cfg}")

        device_text = str(tracker.get("device", "cpu")).strip().lower()
        provider = "cuda" if "cuda" in device_text or device_text.startswith("gpu") else "cpu"
        batch_mode = bool(tracker.get("batch_mode", True))

        profile = str(action_mapping.get("profile", "")).strip().lower()
        joint_order = str(action_mapping.get("joint_order", "isaaclab")).strip().lower()
        clamp_raw_action = bool(action_mapping.get("clamp_action", False))
        action_clip = float(action_mapping.get("action_clip", 1.0))

        default_raw = action_mapping.get("default_angles", None)
        if isinstance(default_raw, list) and len(default_raw) >= target_dim:
            default_angles = [float(v) for v in default_raw[:target_dim]]
        elif profile == "sonic_g1_policy_parameters":
            default_angles = [float(v) for v in G1_DEFAULT_JOINT_ANGLES[:target_dim]]
        else:
            default_angles = [0.0 for _ in range(target_dim)]

        scale_raw = action_mapping.get("action_scale", 1.0)
        per_joint_scale_raw = action_mapping.get("per_joint_action_scale", None)

        if isinstance(per_joint_scale_raw, list) and len(per_joint_scale_raw) >= target_dim:
            action_scale = [float(v) for v in per_joint_scale_raw[:target_dim]]
        elif isinstance(scale_raw, list) and len(scale_raw) >= target_dim:
            action_scale = [float(v) for v in scale_raw[:target_dim]]
        elif profile == "sonic_g1_policy_parameters":
            global_scale = float(scale_raw)
            action_scale = [float(v) * global_scale for v in G1_SONIC_ACTION_SCALE[:target_dim]]
        else:
            scalar = float(scale_raw)
            action_scale = [scalar for _ in range(target_dim)]

        return cls(
            encoder_onnx=encoder_onnx,
            decoder_onnx=decoder_onnx,
            observation_config=observation_config,
            obs_mapping_cfg=obs_mapping_cfg,
            provider=provider,
            target_dim=int(target_dim),
            batch_mode=batch_mode,
            joint_order=joint_order,
            clamp_raw_action=clamp_raw_action,
            action_clip=action_clip,
            action_scale=action_scale,
            default_angles=default_angles,
        )


class SonicPythonRuntime:
    """In-process SONIC ONNX runtime aligned with C++ deploy action semantics."""

    def __init__(self, cfg: SonicPythonRuntimeConfig) -> None:
        self.cfg = cfg

        self.adapter = SonicFrozenTrackerAdapter(
            encoder_path=cfg.encoder_onnx,
            decoder_path=cfg.decoder_onnx,
            observation_config_path=cfg.observation_config,
            target_dim=cfg.target_dim,
            obs_mapping_path=cfg.obs_mapping_cfg,
            provider=cfg.provider,
        )

        self._action_scale = torch.tensor(cfg.action_scale, dtype=torch.float32)
        self._default_angles = torch.tensor(cfg.default_angles, dtype=torch.float32)
        self._mujoco_to_isaac_idx = torch.tensor(MUJOCO_TO_ISAACLAB[: cfg.target_dim], dtype=torch.long)

    @classmethod
    def from_tracker_config(
        cls,
        tracker_cfg_path: Path,
        *,
        target_dim: int,
        fallback_dir: Path | None = None,
    ) -> "SonicPythonRuntime":
        cfg = SonicPythonRuntimeConfig.from_tracker_config(
            tracker_cfg_path=tracker_cfg_path,
            target_dim=target_dim,
            fallback_dir=fallback_dir,
        )
        return cls(cfg)

    def reset_env(self, env_id: int) -> None:
        self.adapter.reset_env(env_id)

    def reset_all(self, num_envs: int) -> None:
        for env_id in range(int(num_envs)):
            self.reset_env(env_id)

    def raw_to_absolute_target(self, raw_action: torch.Tensor, *, device: torch.device | str) -> tuple[torch.Tensor, torch.Tensor]:
        raw = map_sonic_output_to_targets(raw_action.flatten().to(dtype=torch.float32), self.cfg.target_dim)

        if self.cfg.joint_order == "mujoco":
            raw = raw[self._mujoco_to_isaac_idx.to(raw.device)]

        if self.cfg.clamp_raw_action:
            raw = torch.clamp(raw, -self.cfg.action_clip, self.cfg.action_clip)

        scale = self._action_scale.to(device=raw.device)
        default = self._default_angles.to(device=raw.device)
        q_target_abs = default + raw * scale

        return raw.to(device=device), q_target_abs.to(device=device)

    def raw_batch_to_absolute_target(
        self,
        raw_action_batch: torch.Tensor,
        *,
        device: torch.device | str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        raw = raw_action_batch.detach().to(dtype=torch.float32)
        if raw.ndim == 1:
            raw = raw.unsqueeze(0)
        if raw.ndim != 2:
            raise ValueError(f"raw_action_batch must be rank-1/2, got shape={tuple(raw.shape)}")

        n = int(raw.shape[0])
        if raw.shape[1] < self.cfg.target_dim:
            padded = torch.zeros((n, self.cfg.target_dim), dtype=torch.float32, device=raw.device)
            padded[:, : raw.shape[1]] = raw
            raw = padded
        else:
            raw = raw[:, : self.cfg.target_dim]

        if self.cfg.joint_order == "mujoco":
            raw = raw[:, self._mujoco_to_isaac_idx.to(raw.device)]

        if self.cfg.clamp_raw_action:
            raw = torch.clamp(raw, -self.cfg.action_clip, self.cfg.action_clip)

        scale = self._action_scale.to(device=raw.device).unsqueeze(0)
        default = self._default_angles.to(device=raw.device).unsqueeze(0)
        q_target_abs = default + raw * scale
        return raw.to(device=device), q_target_abs.to(device=device)

    def infer_one(
        self,
        *,
        current: Observation,
        reference: ReferenceTarget | None,
        reference_generator: Any | None,
        env_id: int,
        return_debug: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, Any] | None]:
        if return_debug:
            raw_action, debug = self.adapter.act_from_structured_debug(
                current=current,
                history=None,
                reference=reference,
                reference_generator=reference_generator,
                env_id=env_id,
            )
        else:
            raw_action = self.adapter.act_from_structured(
                current=current,
                history=None,
                reference=reference,
                reference_generator=reference_generator,
                env_id=env_id,
            )
            debug = None

        raw_isaac, q_target_abs = self.raw_to_absolute_target(raw_action, device=current.q.device)

        if debug is not None:
            debug["raw_action_isaac"] = raw_isaac.detach().clone()
            debug["q_target_abs"] = q_target_abs.detach().clone()

        return raw_isaac, q_target_abs, debug

    def infer_batch(
        self,
        *,
        currents: Sequence[Observation],
        references: Sequence[ReferenceTarget | None] | None,
        reference_generator: Any | None,
        env_ids: Sequence[int] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if len(currents) == 0:
            empty = torch.zeros((0, self.cfg.target_dim), dtype=torch.float32)
            return empty, empty

        if self.cfg.batch_mode:
            raw_batch = self.adapter.act_batch_from_structured(
                currents=currents,
                references=references,
                reference_generator=reference_generator,
                env_ids=env_ids,
            )
        else:
            if env_ids is None:
                env_ids = list(range(len(currents)))
            ref_list = list(references) if references is not None else [None] * len(currents)
            if len(ref_list) != len(currents):
                raise ValueError(f"references length {len(ref_list)} does not match currents length {len(currents)}")
            rows = []
            for i, env_id in enumerate(env_ids):
                rows.append(
                    self.adapter.act_from_structured(
                        current=currents[i],
                        history=None,
                        reference=ref_list[i],
                        reference_generator=reference_generator,
                        env_id=int(env_id),
                    )
                )
            raw_batch = torch.stack(rows, dim=0)
        device = currents[0].q.device
        return self.raw_batch_to_absolute_target(raw_batch, device=device)
