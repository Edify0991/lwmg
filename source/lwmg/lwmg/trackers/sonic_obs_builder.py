from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable
import re

import torch
import yaml

from lwmg.envs.observations import Observation
from lwmg.references.reference_types import ReferenceTarget
from lwmg.sonic_io.sonic_config_parser import parse_observation_config


@dataclass
class SonicObservationPack:
    decoder_obs: torch.Tensor
    encoder_obs: torch.Tensor | None
    token_slice: tuple[int, int] | None


@dataclass
class _AnchorHeadingContext:
    base_quat: torch.Tensor
    apply_delta_heading: torch.Tensor


@dataclass(frozen=True)
class _FeatureBuildInputs:
    current: Observation
    history: Iterable[Observation] | None
    reference: ReferenceTarget | None
    reference_generator: Any | None
    anchor_ctx: _AnchorHeadingContext


@dataclass(frozen=True)
class _ObservationRegistryEntry:
    """Centralized SONIC observation registry entry (official C++ style)."""

    name: str
    pattern: re.Pattern[str]
    dim_fn: Callable[[re.Match[str], int], int]
    gather_fn: Callable[[re.Match[str], _FeatureBuildInputs], torch.Tensor]


@dataclass
class SonicObservationBuilder:
    """Build SONIC decoder/encoder observation vectors from Isaac Lab observations."""

    mapping_path: Path
    observation_config_path: Path
    debug: bool = False

    _heading_init_base_quat_by_env: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _heading_init_ref_root_quat_by_env: dict[int, torch.Tensor] = field(default_factory=dict, init=False, repr=False)
    _heading_last_ref_frame_by_env: dict[int, int] = field(default_factory=dict, init=False, repr=False)
    _heading_delta_by_env: dict[int, float] = field(default_factory=dict, init=False, repr=False)
    _observation_registry: list[_ObservationRegistryEntry] = field(default_factory=list, init=False, repr=False)

    def __post_init__(self) -> None:
        mapping_raw = yaml.safe_load(self.mapping_path.read_text()) or {}
        self.mapping: dict[str, str] = mapping_raw.get("mapping", {})
        self.mapping_raw = mapping_raw

        if self.observation_config_path.exists():
            self.obs_cfg_raw = yaml.safe_load(self.observation_config_path.read_text()) or {}
        else:
            self.obs_cfg_raw = {}

        self.obs_cfg = parse_observation_config(self.observation_config_path)
        if not bool(self.obs_cfg.get("parse_ok", True)):
            err = str(self.obs_cfg.get("parse_error", "SONIC observation config parsing failed")).strip()
            raise ValueError(f"Invalid SONIC observation config: {err}")

        requested = int(self.obs_cfg.get("history_steps", 1))
        self.history_steps = 4 if requested >= 4 else 1

        self.strict_mode = bool(self.obs_cfg.get("is_official_format", False))

        robot_cfg = mapping_raw.get("robot", {})
        self.lower_body_indices = list(
            robot_cfg.get(
                "lower_body_joint_indices_mujoco_order_in_isaaclab_index",
                [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18],
            )
        )
        self.wrist_indices = list(
            robot_cfg.get(
                "wrist_joint_indices_isaaclab_order_in_isaaclab_index",
                [23, 24, 25, 26, 27, 28],
            )
        )

        sonic_cfg = mapping_raw.get("sonic", {})
        self.strict_dim_check = bool(sonic_cfg.get("strict_dim_check", True))
        self.fail_on_missing_key = bool(sonic_cfg.get("fail_on_missing_key", True))

        runtime_cfg = mapping_raw.get("runtime", {})
        self.enforce_g1_mode = bool(runtime_cfg.get("enforce_g1_mode", True))
        requested_mode_name = str(runtime_cfg.get("encoder_mode_name", "g1")).strip().lower()
        requested_mode_id_raw = runtime_cfg.get("encoder_mode", None)

        self.encoder_mode_name = requested_mode_name
        try:
            self.encoder_mode = int(requested_mode_id_raw) if requested_mode_id_raw is not None else 0
        except (TypeError, ValueError):
            self.encoder_mode = 0

        self.decoder_obs_names: list[str] = []
        self.encoder_obs_names: list[str] = []
        self.encoder_dim = 0
        self.expected_decoder_dim = None
        self.expected_encoder_dim = None
        self.token_slice: tuple[int, int] | None = None
        self.encoder_mode_to_required: dict[int, set[str]] = {}
        self.encoder_required_observations: set[str] | None = None

        if self.strict_mode:
            self.decoder_obs_names = [str(name) for name in self.obs_cfg.get("decoder_observations", [])]
            self.encoder_dim = int(self.obs_cfg.get("encoder_dimension", 0))
            self.encoder_obs_names = [str(name) for name in self.obs_cfg.get("encoder_observations", [])]

            mode_name_to_id_raw = self.obs_cfg.get("encoder_mode_name_to_id", {})
            if not isinstance(mode_name_to_id_raw, dict):
                mode_name_to_id_raw = {}
            mode_name_to_id = {
                str(k).strip().lower(): int(v)
                for k, v in mode_name_to_id_raw.items()
                if str(k).strip()
            }

            mode_id_to_required_raw = self.obs_cfg.get("encoder_mode_id_to_required", {})
            if not isinstance(mode_id_to_required_raw, dict):
                mode_id_to_required_raw = {}
            for mode_id_raw, required_raw in mode_id_to_required_raw.items():
                try:
                    mode_id = int(mode_id_raw)
                except (TypeError, ValueError):
                    continue
                if not isinstance(required_raw, list):
                    required_raw = []
                self.encoder_mode_to_required[mode_id] = {
                    str(item) for item in required_raw if item is not None and str(item).strip()
                }

            g1_mode_id = self.obs_cfg.get("g1_mode_id", None)
            g1_mode_id = int(g1_mode_id) if g1_mode_id is not None else None

            has_encoder_modes = bool(self.encoder_mode_to_required)
            if self.enforce_g1_mode and has_encoder_modes:
                if g1_mode_id is None:
                    raise ValueError(
                        "SONIC observation config does not define encoder mode 'g1', but runtime.enforce_g1_mode=true"
                    )
                if requested_mode_name not in {"", "g1"}:
                    raise ValueError(
                        f"runtime.encoder_mode_name='{requested_mode_name}' is not allowed when enforce_g1_mode=true"
                    )
                if requested_mode_id_raw is not None and int(requested_mode_id_raw) != g1_mode_id:
                    raise ValueError(
                        f"runtime.encoder_mode={requested_mode_id_raw} does not match g1 mode id={g1_mode_id}"
                    )
                self.encoder_mode = g1_mode_id
                self.encoder_mode_name = "g1"
            else:
                if requested_mode_id_raw is not None:
                    self.encoder_mode = int(requested_mode_id_raw)
                elif requested_mode_name and requested_mode_name in mode_name_to_id:
                    self.encoder_mode = mode_name_to_id[requested_mode_name]
                elif g1_mode_id is not None:
                    self.encoder_mode = g1_mode_id
                self.encoder_mode_name = requested_mode_name

            self._observation_registry = self._build_observation_registry()
            self.encoder_required_observations = self.encoder_mode_to_required.get(self.encoder_mode)

            self.expected_decoder_dim = sum(self._feature_dim(name, token_dim=self.encoder_dim) for name in self.decoder_obs_names)
            self.expected_encoder_dim = sum(self._feature_dim(name, token_dim=self.encoder_dim) for name in self.encoder_obs_names)

            offset = 0
            for name in self.decoder_obs_names:
                d = self._feature_dim(name, token_dim=self.encoder_dim)
                if name == "token_state":
                    self.token_slice = (offset, offset + d)
                offset += d


    def _build_feature_offset_rows(self, names: list[str], stream: str) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        offset = 0
        for name in names:
            dim = int(self._feature_dim(name, token_dim=self.encoder_dim))
            row: dict[str, Any] = {
                "name": name,
                "start": int(offset),
                "end": int(offset + dim),
                "dim": int(dim),
            }
            if stream == "encoder":
                required = self.encoder_required_observations is None or name in self.encoder_required_observations
                row["required"] = bool(required)
            rows.append(row)
            offset += dim
        return rows

    def feature_layout(self) -> dict[str, Any]:
        """Return per-feature [start:end) layout for decoder/encoder vectors."""
        if not self.strict_mode:
            return {
                "strict_mode": False,
                "decoder": [],
                "encoder": [],
                "decoder_total_dim": 0,
                "encoder_total_dim": 0,
                "token_slice": None,
                "encoder_mode": self.encoder_mode,
                "encoder_mode_name": self.encoder_mode_name,
            }

        decoder_rows = self._build_feature_offset_rows(self.decoder_obs_names, stream="decoder")
        encoder_rows = self._build_feature_offset_rows(self.encoder_obs_names, stream="encoder")

        decoder_total = 0 if not decoder_rows else int(decoder_rows[-1]["end"])
        encoder_total = 0 if not encoder_rows else int(encoder_rows[-1]["end"])

        return {
            "strict_mode": True,
            "decoder": decoder_rows,
            "encoder": encoder_rows,
            "decoder_total_dim": decoder_total,
            "encoder_total_dim": encoder_total,
            "token_slice": self.token_slice,
            "encoder_mode": int(self.encoder_mode),
            "encoder_mode_name": self.encoder_mode_name,
        }

    def reset_env(self, env_id: int) -> None:
        self._heading_init_base_quat_by_env.pop(env_id, None)
        self._heading_init_ref_root_quat_by_env.pop(env_id, None)
        self._heading_last_ref_frame_by_env.pop(env_id, None)
        self._heading_delta_by_env.pop(env_id, None)

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

    def _legacy_build(self, current: Observation, history: Iterable[Observation] | None = None) -> torch.Tensor:
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
        return out

    def build(
        self,
        current: Observation,
        history: Iterable[Observation] | None = None,
        reference: ReferenceTarget | None = None,
        reference_generator: Any | None = None,
        env_id: int = 0,
    ) -> torch.Tensor:
        if not self.strict_mode:
            return self._legacy_build(current=current, history=history)
        return self.build_pack(
            current=current,
            history=history,
            reference=reference,
            reference_generator=reference_generator,
            env_id=env_id,
        ).decoder_obs

    def build_pack(
        self,
        current: Observation,
        history: Iterable[Observation] | None = None,
        reference: ReferenceTarget | None = None,
        reference_generator: Any | None = None,
        env_id: int = 0,
    ) -> SonicObservationPack:
        if not self.strict_mode:
            legacy = self._legacy_build(current=current, history=history)
            return SonicObservationPack(decoder_obs=legacy, encoder_obs=None, token_slice=None)

        anchor_ctx = self._prepare_anchor_heading_context(
            env_id=env_id,
            current=current,
            reference=reference,
        )

        decoder_chunks = [
            self._gather_feature(
                name=name,
                current=current,
                history=history,
                reference=reference,
                reference_generator=reference_generator,
                anchor_ctx=anchor_ctx,
            )
            for name in self.decoder_obs_names
        ]
        decoder_obs = torch.cat(decoder_chunks, dim=0).to(dtype=torch.float32)
        encoder_chunks: list[torch.Tensor] = []
        for name in self.encoder_obs_names:
            if self.encoder_required_observations is not None and name not in self.encoder_required_observations:
                encoder_chunks.append(
                    torch.zeros(
                        self._feature_dim(name, token_dim=self.encoder_dim),
                        dtype=torch.float32,
                        device=current.q.device,
                    )
                )
                continue

            encoder_chunks.append(
                self._gather_feature(
                    name=name,
                    current=current,
                    history=history,
                    reference=reference,
                    reference_generator=reference_generator,
                    anchor_ctx=anchor_ctx,
                )
            )

        encoder_obs = torch.cat(encoder_chunks, dim=0).to(dtype=torch.float32) if encoder_chunks else None

        if self.strict_dim_check:
            if self.expected_decoder_dim is not None and decoder_obs.numel() != self.expected_decoder_dim:
                raise ValueError(
                    f"Decoder obs dim mismatch: expected {self.expected_decoder_dim}, got {decoder_obs.numel()}"
                )
            if encoder_obs is not None and self.expected_encoder_dim is not None and encoder_obs.numel() != self.expected_encoder_dim:
                raise ValueError(
                    f"Encoder obs dim mismatch: expected {self.expected_encoder_dim}, got {encoder_obs.numel()}"
                )

        if self.debug:
            required = self.encoder_required_observations
            required_count = -1 if required is None else len(required)
            print(
                f"[SonicObservationBuilder] strict decoder_dim={decoder_obs.numel()} "
                f"encoder_dim={0 if encoder_obs is None else encoder_obs.numel()} "
                f"encoder_mode={self.encoder_mode} required_count={required_count}"
            )

        return SonicObservationPack(decoder_obs=decoder_obs, encoder_obs=encoder_obs, token_slice=self.token_slice)

    def _build_observation_registry(self) -> list[_ObservationRegistryEntry]:
        base_dims = {
            "base_angular_velocity": 3,
            "body_joint_positions": 29,
            "body_joint_velocities": 29,
            "last_actions": 29,
            "gravity_dir": 3,
        }

        def _exact(name: str) -> re.Pattern[str]:
            return re.compile(rf"^{re.escape(name)}$")

        def _zero_like(dim: int, inputs: _FeatureBuildInputs) -> torch.Tensor:
            return torch.zeros(dim, dtype=torch.float32, device=inputs.current.q.device)

        def _single_history(base_name: str, inputs: _FeatureBuildInputs) -> torch.Tensor:
            cur = inputs.current
            if base_name == "base_angular_velocity":
                return cur.imu_gyro.flatten()
            if base_name == "body_joint_positions":
                return cur.q.flatten()
            if base_name == "body_joint_velocities":
                return cur.dq.flatten()
            if base_name == "last_actions":
                return cur.prev_action.flatten()
            return self._gravity_dir(frame=cur).flatten()

        def _sampled_history(match: re.Match[str], inputs: _FeatureBuildInputs) -> torch.Tensor:
            base_name = match.group("base")
            num_frames = int(match.group("frames"))
            step = int(match.group("step"))
            frames = self._history_frames(
                current=inputs.current,
                history=inputs.history,
                num_frames=num_frames,
                step=step,
            )
            if base_name == "base_angular_velocity":
                chunks = [f.imu_gyro.flatten() for f in frames]
            elif base_name == "body_joint_positions":
                chunks = [f.q.flatten() for f in frames]
            elif base_name == "body_joint_velocities":
                chunks = [f.dq.flatten() for f in frames]
            elif base_name == "last_actions":
                chunks = [f.prev_action.flatten() for f in frames]
            else:
                chunks = [self._gravity_dir(frame=f).flatten() for f in frames]
            return torch.cat(chunks, dim=0).to(dtype=torch.float32)

        def _motion_joint_dim(match: re.Match[str], _token_dim: int) -> int:
            part = match.group("part")
            frames = match.group("frames")
            if part == "lowerbody":
                one_frame_dim = len(self.lower_body_indices)
            elif part == "wrists":
                one_frame_dim = len(self.wrist_indices)
            else:
                one_frame_dim = 29

            if frames is None:
                return one_frame_dim
            return int(frames) * one_frame_dim

        def _motion_joint_gather(match: re.Match[str], inputs: _FeatureBuildInputs) -> torch.Tensor:
            kind = match.group("kind")
            part = match.group("part")
            frames = match.group("frames")
            step = match.group("step")

            if kind == "positions":
                if frames is None:
                    vec = self._current_joint_pos_ref(reference=inputs.reference, current=inputs.current)
                else:
                    vec = self._motion_joint_pos_window(
                        num_frames=int(frames),
                        step=int(step),
                        reference=inputs.reference,
                        reference_generator=inputs.reference_generator,
                        current=inputs.current,
                    )
            else:
                if frames is None:
                    vec = self._current_joint_vel_ref(reference=inputs.reference, current=inputs.current)
                else:
                    vec = self._motion_joint_vel_window(
                        num_frames=int(frames),
                        step=int(step),
                        reference=inputs.reference,
                        reference_generator=inputs.reference_generator,
                        current=inputs.current,
                    )

            if part == "lowerbody":
                vec = vec[..., self.lower_body_indices]
            elif part == "wrists":
                vec = vec[..., self.wrist_indices]

            return vec.flatten().to(dtype=torch.float32)

        def _motion_anchor_orientation_window(match: re.Match[str], inputs: _FeatureBuildInputs) -> torch.Tensor:
            n = int(match.group("frames"))
            step = int(match.group("step"))
            quats = self._motion_root_quat_window(
                num_frames=n,
                step=step,
                reference=inputs.reference,
                reference_generator=inputs.reference_generator,
                current=inputs.current,
            )
            chunks = [
                self._anchor_orientation(
                    base_quat=inputs.anchor_ctx.base_quat,
                    ref_quat=self._apply_heading_to_ref_quat(ref_quat=q, anchor_ctx=inputs.anchor_ctx),
                )
                for q in quats
            ]
            return torch.cat(chunks, dim=0).to(dtype=torch.float32)

        return [
            _ObservationRegistryEntry(
                name="token_state",
                pattern=_exact("token_state"),
                dim_fn=lambda _m, token_dim: token_dim,
                gather_fn=lambda _m, inputs: _zero_like(self.encoder_dim, inputs),
            ),
            _ObservationRegistryEntry(
                name="encoder_mode",
                pattern=_exact("encoder_mode"),
                dim_fn=lambda _m, _token_dim: 3,
                gather_fn=lambda _m, inputs: torch.tensor(
                    [float(self.encoder_mode), 0.0, 0.0],
                    dtype=torch.float32,
                    device=inputs.current.q.device,
                ),
            ),
            _ObservationRegistryEntry(
                name="encoder_mode_4",
                pattern=_exact("encoder_mode_4"),
                dim_fn=lambda _m, _token_dim: 4,
                gather_fn=lambda _m, inputs: torch.tensor(
                    [float(self.encoder_mode), 0.0, 0.0, 0.0],
                    dtype=torch.float32,
                    device=inputs.current.q.device,
                ),
            ),
            _ObservationRegistryEntry(
                name="vr_3point_local_target",
                pattern=_exact("vr_3point_local_target"),
                dim_fn=lambda _m, _token_dim: 9,
                gather_fn=lambda _m, inputs: _zero_like(9, inputs),
            ),
            _ObservationRegistryEntry(
                name="vr_3point_local_orn_target",
                pattern=_exact("vr_3point_local_orn_target"),
                dim_fn=lambda _m, _token_dim: 12,
                gather_fn=lambda _m, inputs: _zero_like(12, inputs),
            ),
            _ObservationRegistryEntry(
                name="smpl_joints",
                pattern=re.compile(r"^smpl_joints(?:_(?P<frames>\d+)frame_step(?P<step>\d+))?$") ,
                dim_fn=lambda m, _token_dim: (int(m.group("frames")) if m.group("frames") else 1) * 72,
                gather_fn=lambda m, inputs: _zero_like(
                    (int(m.group("frames")) if m.group("frames") else 1) * 72,
                    inputs,
                ),
            ),
            _ObservationRegistryEntry(
                name="smpl_anchor_orientation",
                pattern=re.compile(r"^smpl_anchor_orientation(?:_(?P<frames>\d+)frame_step(?P<step>\d+))?$") ,
                dim_fn=lambda m, _token_dim: (int(m.group("frames")) if m.group("frames") else 1) * 6,
                gather_fn=lambda m, inputs: _zero_like(
                    (int(m.group("frames")) if m.group("frames") else 1) * 6,
                    inputs,
                ),
            ),
            _ObservationRegistryEntry(
                name="history_base",
                pattern=re.compile(
                    r"^(?P<base>base_angular_velocity|body_joint_positions|body_joint_velocities|last_actions|gravity_dir)$"
                ),
                dim_fn=lambda m, _token_dim: base_dims[m.group("base")],
                gather_fn=lambda m, inputs: _single_history(m.group("base"), inputs).to(dtype=torch.float32),
            ),
            _ObservationRegistryEntry(
                name="history_window",
                pattern=re.compile(
                    r"^his_(?P<base>base_angular_velocity|body_joint_positions|body_joint_velocities|last_actions|gravity_dir)_(?P<frames>\d+)frame_step(?P<step>\d+)$"
                ),
                dim_fn=lambda m, _token_dim: int(m.group("frames")) * base_dims[m.group("base")],
                gather_fn=_sampled_history,
            ),
            _ObservationRegistryEntry(
                name="motion_joint",
                pattern=re.compile(
                    r"^motion_joint_(?P<kind>positions|velocities)(?:_(?P<part>lowerbody|wrists))?(?:_(?P<frames>\d+)frame_step(?P<step>\d+))?$"
                ),
                dim_fn=_motion_joint_dim,
                gather_fn=_motion_joint_gather,
            ),
            _ObservationRegistryEntry(
                name="motion_root_z_position",
                pattern=_exact("motion_root_z_position"),
                dim_fn=lambda _m, _token_dim: 1,
                gather_fn=lambda _m, inputs: self._current_root_pos(reference=inputs.reference, current=inputs.current)[2:3].to(
                    dtype=torch.float32
                ),
            ),
            _ObservationRegistryEntry(
                name="motion_root_z_position_window",
                pattern=re.compile(r"^motion_root_z_position_(?P<frames>\d+)frame_step(?P<step>\d+)$"),
                dim_fn=lambda m, _token_dim: int(m.group("frames")),
                gather_fn=lambda m, inputs: self._motion_root_pos_window(
                    num_frames=int(m.group("frames")),
                    step=int(m.group("step")),
                    reference=inputs.reference,
                    reference_generator=inputs.reference_generator,
                    current=inputs.current,
                )[:, 2]
                .flatten()
                .to(dtype=torch.float32),
            ),
            _ObservationRegistryEntry(
                name="motion_anchor_orientation",
                pattern=_exact("motion_anchor_orientation"),
                dim_fn=lambda _m, _token_dim: 6,
                gather_fn=lambda _m, inputs: self._anchor_orientation(
                    base_quat=inputs.anchor_ctx.base_quat,
                    ref_quat=self._apply_heading_to_ref_quat(
                        ref_quat=self._current_root_quat(reference=inputs.reference, current=inputs.current),
                        anchor_ctx=inputs.anchor_ctx,
                    ),
                ),
            ),
            _ObservationRegistryEntry(
                name="motion_anchor_orientation_window",
                pattern=re.compile(r"^motion_anchor_orientation_(?P<frames>\d+)frame_step(?P<step>\d+)$"),
                dim_fn=lambda m, _token_dim: int(m.group("frames")) * 6,
                gather_fn=_motion_anchor_orientation_window,
            ),
        ]

    def _resolve_observation_registry_entry(
        self, name: str
    ) -> tuple[_ObservationRegistryEntry, re.Match[str]] | None:
        for entry in self._observation_registry:
            match = entry.pattern.match(name)
            if match is not None:
                return entry, match
        return None

    def _gather_feature(
        self,
        name: str,
        current: Observation,
        history: Iterable[Observation] | None,
        reference: ReferenceTarget | None,
        reference_generator: Any | None,
        anchor_ctx: _AnchorHeadingContext,
    ) -> torch.Tensor:
        resolved = self._resolve_observation_registry_entry(name)
        if resolved is not None:
            entry, match = resolved
            inputs = _FeatureBuildInputs(
                current=current,
                history=history,
                reference=reference,
                reference_generator=reference_generator,
                anchor_ctx=anchor_ctx,
            )
            return entry.gather_fn(match, inputs).to(dtype=torch.float32)

        if self.fail_on_missing_key:
            raise KeyError(f"Unsupported SONIC observation feature: {name}")
        return torch.zeros(self._feature_dim(name, token_dim=self.encoder_dim), dtype=torch.float32, device=current.q.device)

    def _feature_dim(self, name: str, token_dim: int) -> int:
        resolved = self._resolve_observation_registry_entry(name)
        if resolved is not None:
            entry, match = resolved
            return int(entry.dim_fn(match, token_dim))

        if self.fail_on_missing_key:
            raise KeyError(f"Unknown feature dim for SONIC observation '{name}'")
        return 0

    def _extract_frames(self, name: str) -> int:
        m = re.search(r"_(\d+)frame_step\d+$", name)
        if not m:
            return 1
        return int(m.group(1))

    def _history_frames(
        self,
        current: Observation,
        history: Iterable[Observation] | None,
        num_frames: int,
        step: int,
    ) -> list[Observation]:
        if num_frames <= 0:
            return []

        hist = list(history or [])
        available = hist + [current]
        newest_idx = len(available) - 1

        sampled_rev: list[Observation] = []
        for i in range(num_frames):
            idx = newest_idx - i * step
            if idx >= 0:
                sampled_rev.append(available[idx])
            else:
                sampled_rev.append(self._zero_observation_like(current))

        sampled_rev.reverse()  # SONIC GetLatest(..., newest_first=false) => oldest -> newest
        return sampled_rev

    def _zero_observation_like(self, current: Observation) -> Observation:
        q = torch.zeros_like(current.q)
        dq = torch.zeros_like(current.dq)
        imu_accel = torch.zeros_like(current.imu_accel)
        imu_gyro = torch.zeros_like(current.imu_gyro)
        prev_action = torch.zeros_like(current.prev_action)
        contacts = torch.zeros_like(current.contacts)
        tracking_error = torch.zeros_like(current.tracking_error_summary)

        root_quat = None
        if current.root_quat_wxyz is not None:
            root_quat = torch.zeros_like(current.root_quat_wxyz)

        projected_gravity = None
        if current.projected_gravity is not None:
            projected_gravity = torch.zeros_like(current.projected_gravity)

        return Observation(
            q=q,
            dq=dq,
            imu_accel=imu_accel,
            imu_gyro=imu_gyro,
            prev_action=prev_action,
            contacts=contacts,
            tracking_error_summary=tracking_error,
            root_quat_wxyz=root_quat,
            projected_gravity=projected_gravity,
        )

    def _motion_window(
        self,
        feature: str,
        name: str,
        current: Observation,
        reference: ReferenceTarget | None,
        reference_generator: Any | None,
    ) -> torch.Tensor:
        m = re.match(r"^motion_joint_(positions|velocities)$", name)
        if m:
            if feature == "joint_pos":
                return self._current_joint_pos_ref(reference=reference, current=current).to(dtype=torch.float32)
            return self._current_joint_vel_ref(reference=reference, current=current).to(dtype=torch.float32)

        m = re.match(r"^motion_joint_(positions|velocities)_(lowerbody|wrists)?_?(\d+)frame_step(\d+)$", name)
        if not m:
            if self.fail_on_missing_key:
                raise KeyError(f"Unsupported motion window feature '{name}'")
            return torch.zeros(0, dtype=torch.float32, device=current.q.device)

        part = m.group(2)
        n = int(m.group(3))
        step = int(m.group(4))

        if feature == "joint_pos":
            window = self._motion_joint_pos_window(n, step, reference, reference_generator, current)
        else:
            window = self._motion_joint_vel_window(n, step, reference, reference_generator, current)

        if part == "lowerbody":
            window = window[:, self.lower_body_indices]
        elif part == "wrists":
            window = window[:, self.wrist_indices]

        return window.flatten().to(dtype=torch.float32)

    def _motion_joint_pos_window(
        self,
        num_frames: int,
        step: int,
        reference: ReferenceTarget | None,
        reference_generator: Any | None,
        current: Observation,
    ) -> torch.Tensor:
        start_idx = self._frame_index(reference)
        if reference_generator is not None and hasattr(reference_generator, "window_joint_pos"):
            return reference_generator.window_joint_pos(num_frames=num_frames, step=step, start_idx=start_idx)

        cur = self._current_joint_pos_ref(reference=reference, current=current)
        return cur.unsqueeze(0).repeat(num_frames, 1)

    def _motion_joint_vel_window(
        self,
        num_frames: int,
        step: int,
        reference: ReferenceTarget | None,
        reference_generator: Any | None,
        current: Observation,
    ) -> torch.Tensor:
        start_idx = self._frame_index(reference)
        if reference_generator is not None and hasattr(reference_generator, "window_joint_vel"):
            return reference_generator.window_joint_vel(num_frames=num_frames, step=step, start_idx=start_idx)

        cur = self._current_joint_vel_ref(reference=reference, current=current)
        return cur.unsqueeze(0).repeat(num_frames, 1)

    def _motion_root_pos_window(
        self,
        num_frames: int,
        step: int,
        reference: ReferenceTarget | None,
        reference_generator: Any | None,
        current: Observation,
    ) -> torch.Tensor:
        start_idx = self._frame_index(reference)
        if reference_generator is not None and hasattr(reference_generator, "window_root_pos"):
            return reference_generator.window_root_pos(num_frames=num_frames, step=step, start_idx=start_idx)

        cur = self._current_root_pos(reference=reference, current=current)
        return cur.unsqueeze(0).repeat(num_frames, 1)

    def _motion_root_quat_window(
        self,
        num_frames: int,
        step: int,
        reference: ReferenceTarget | None,
        reference_generator: Any | None,
        current: Observation,
    ) -> torch.Tensor:
        start_idx = self._frame_index(reference)
        if reference_generator is not None and hasattr(reference_generator, "window_root_quat"):
            return reference_generator.window_root_quat(num_frames=num_frames, step=step, start_idx=start_idx)

        cur = self._current_root_quat(reference=reference, current=current)
        return cur.unsqueeze(0).repeat(num_frames, 1)

    def _frame_index(self, reference: ReferenceTarget | None) -> int | None:
        if reference is None or reference.frame_idx is None or reference.frame_idx.numel() == 0:
            return None
        return int(reference.frame_idx.flatten()[0].item())

    def _current_joint_pos_ref(self, reference: ReferenceTarget | None, current: Observation) -> torch.Tensor:
        if reference is not None and reference.joint_pos.numel() > 0:
            return reference.joint_pos[0].to(device=current.q.device, dtype=torch.float32)
        return torch.zeros_like(current.q)

    def _current_joint_vel_ref(self, reference: ReferenceTarget | None, current: Observation) -> torch.Tensor:
        if reference is not None and reference.joint_vel.numel() > 0:
            return reference.joint_vel[0].to(device=current.q.device, dtype=torch.float32)
        return torch.zeros_like(current.dq)

    def _current_root_pos(self, reference: ReferenceTarget | None, current: Observation) -> torch.Tensor:
        if reference is not None and reference.root_pos is not None and reference.root_pos.numel() > 0:
            return reference.root_pos[0].to(device=current.q.device, dtype=torch.float32)
        return torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device=current.q.device)

    def _current_root_quat(self, reference: ReferenceTarget | None, current: Observation) -> torch.Tensor:
        if reference is not None and reference.root_quat is not None and reference.root_quat.numel() > 0:
            return self._quat_normalize(reference.root_quat[0].to(device=current.q.device, dtype=torch.float32))
        return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=current.q.device)

    def _base_quat(self, current: Observation) -> torch.Tensor:
        if current.root_quat_wxyz is None:
            return torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=current.q.device)
        return self._quat_normalize(current.root_quat_wxyz.to(dtype=torch.float32, device=current.q.device))

    def _extract_delta_heading_from_reference(self, reference: ReferenceTarget | None) -> float | None:
        if reference is None or not reference.extras:
            return None

        for key in ["delta_heading", "delta_heading_rad", "heading_delta"]:
            if key not in reference.extras:
                continue
            value = reference.extras[key]
            if isinstance(value, torch.Tensor):
                if value.numel() == 0:
                    continue
                scalar = float(value.flatten()[0].item())
            else:
                try:
                    scalar = float(value)
                except (TypeError, ValueError):
                    continue
            if not torch.isfinite(torch.tensor(scalar)):
                continue
            return scalar
        return None

    def _prepare_anchor_heading_context(
        self,
        env_id: int,
        current: Observation,
        reference: ReferenceTarget | None,
    ) -> _AnchorHeadingContext:
        base_quat = self._base_quat(current)
        ref_quat_now = self._current_root_quat(reference=reference, current=current)
        frame_idx = self._frame_index(reference)
        last_frame = self._heading_last_ref_frame_by_env.get(env_id, None)
        frame_restarted = frame_idx is not None and (
            frame_idx == 0 or (last_frame is not None and frame_idx < last_frame)
        )
        need_reinit = (
            env_id not in self._heading_init_base_quat_by_env
            or env_id not in self._heading_init_ref_root_quat_by_env
            or frame_restarted
        )

        if need_reinit:
            self._heading_init_base_quat_by_env[env_id] = base_quat.detach().clone()
            self._heading_init_ref_root_quat_by_env[env_id] = ref_quat_now.detach().clone()
            # Match official C++ HeadingState reset: delta heading is reset on reinit.
            self._heading_delta_by_env[env_id] = 0.0

        if frame_idx is not None:
            self._heading_last_ref_frame_by_env[env_id] = int(frame_idx)

        delta_heading_update = self._extract_delta_heading_from_reference(reference)
        if delta_heading_update is not None:
            self._heading_delta_by_env[env_id] = float(delta_heading_update)
        delta_heading = float(self._heading_delta_by_env.get(env_id, 0.0))

        init_base_quat = self._heading_init_base_quat_by_env.get(env_id, base_quat)
        init_ref_root_quat = self._heading_init_ref_root_quat_by_env.get(env_id, ref_quat_now)

        init_heading = self._calc_heading_quat(init_base_quat)
        data_heading_inv = self._calc_heading_quat_inv(init_ref_root_quat)
        apply_delta_heading = self._quat_mul(init_heading, data_heading_inv)
        if abs(delta_heading) > 1e-12:
            apply_delta_heading = self._quat_mul(self._euler_z_to_quat(delta_heading, base_quat), apply_delta_heading)

        return _AnchorHeadingContext(
            base_quat=base_quat,
            apply_delta_heading=self._quat_normalize(apply_delta_heading),
        )

    def _apply_heading_to_ref_quat(self, ref_quat: torch.Tensor, anchor_ctx: _AnchorHeadingContext) -> torch.Tensor:
        ref = self._quat_normalize(ref_quat.to(dtype=anchor_ctx.base_quat.dtype, device=anchor_ctx.base_quat.device))
        return self._quat_mul(anchor_ctx.apply_delta_heading, ref)

    def _calc_heading_angle(self, q: torch.Tensor) -> torch.Tensor:
        ref_dir = torch.tensor([1.0, 0.0, 0.0], dtype=q.dtype, device=q.device)
        rot_dir = self._quat_rotate(self._quat_normalize(q), ref_dir)
        return torch.atan2(rot_dir[1], rot_dir[0])

    def _quat_from_angle_axis(self, angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
        axis_norm = axis / torch.clamp(torch.linalg.norm(axis), min=1e-8)
        theta = angle * 0.5
        sin_theta = torch.sin(theta)
        cos_theta = torch.cos(theta)
        q = torch.stack(
            [
                cos_theta,
                axis_norm[0] * sin_theta,
                axis_norm[1] * sin_theta,
                axis_norm[2] * sin_theta,
            ],
            dim=0,
        )
        return self._quat_normalize(q)

    def _calc_heading_quat(self, q: torch.Tensor) -> torch.Tensor:
        heading = self._calc_heading_angle(self._quat_normalize(q))
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)
        return self._quat_from_angle_axis(heading, axis)

    def _calc_heading_quat_inv(self, q: torch.Tensor) -> torch.Tensor:
        heading = self._calc_heading_angle(self._quat_normalize(q))
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=q.dtype, device=q.device)
        return self._quat_from_angle_axis(-heading, axis)

    def _euler_z_to_quat(self, angle_rad: float, like: torch.Tensor) -> torch.Tensor:
        angle = torch.tensor(float(angle_rad), dtype=like.dtype, device=like.device)
        axis = torch.tensor([0.0, 0.0, 1.0], dtype=like.dtype, device=like.device)
        return self._quat_from_angle_axis(angle, axis)

    def _gravity_dir(self, frame: Observation) -> torch.Tensor:
        if frame.projected_gravity is not None:
            return frame.projected_gravity.to(dtype=torch.float32)
        if frame.root_quat_wxyz is not None:
            q_inv = self._quat_conjugate(self._quat_normalize(frame.root_quat_wxyz.to(dtype=torch.float32)))
            return self._quat_rotate(q_inv, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=q_inv.device))
        return torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32, device=frame.q.device)

    def _anchor_orientation(self, base_quat: torch.Tensor, ref_quat: torch.Tensor) -> torch.Tensor:
        base_to_ref = self._quat_mul(self._quat_conjugate(base_quat), self._quat_normalize(ref_quat))
        rot = self._quat_to_matrix(base_to_ref)
        out = torch.stack(
            [
                rot[0, 0],
                rot[0, 1],
                rot[1, 0],
                rot[1, 1],
                rot[2, 0],
                rot[2, 1],
            ],
            dim=0,
        )
        return out.to(dtype=torch.float32)

    def _quat_normalize(self, q: torch.Tensor) -> torch.Tensor:
        return q / torch.clamp(torch.linalg.norm(q), min=1e-8)

    def _quat_conjugate(self, q: torch.Tensor) -> torch.Tensor:
        return torch.stack([q[0], -q[1], -q[2], -q[3]], dim=0)

    def _quat_mul(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        aw, ax, ay, az = a
        bw, bx, by, bz = b
        return torch.stack(
            [
                aw * bw - ax * bx - ay * by - az * bz,
                aw * bx + ax * bw + ay * bz - az * by,
                aw * by - ax * bz + ay * bw + az * bx,
                aw * bz + ax * by - ay * bx + az * bw,
            ],
            dim=0,
        )

    def _quat_to_matrix(self, q: torch.Tensor) -> torch.Tensor:
        w, x, y, z = self._quat_normalize(q)
        return torch.tensor(
            [
                [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
                [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
                [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
            ],
            dtype=q.dtype,
            device=q.device,
        )

    def _quat_rotate(self, q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        vq = torch.stack([torch.tensor(0.0, dtype=q.dtype, device=q.device), v[0], v[1], v[2]], dim=0)
        rotated = self._quat_mul(self._quat_mul(q, vq), self._quat_conjugate(q))
        return rotated[1:]
