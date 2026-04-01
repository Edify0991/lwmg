from __future__ import annotations

from pathlib import Path
import re
from typing import Any, Dict

import yaml


_HISTORY_PATTERN = re.compile(r"_(\d+)frame_step(\d+)$")
_DEFAULT_OBSERVATIONS: tuple[str, ...] = (
    "motion_joint_positions",
    "motion_joint_velocities",
    "motion_anchor_orientation",
    "base_angular_velocity",
    "body_joint_positions",
    "body_joint_velocities",
    "last_actions",
)


def _default_observation_configs() -> list[dict[str, Any]]:
    return [{"name": name, "enabled": True} for name in _DEFAULT_OBSERVATIONS]


def _normalize_observation_configs(items: Any) -> list[dict[str, Any]]:
    if not isinstance(items, list):
        return []

    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str):
            continue
        name = name.strip()
        if not name:
            continue
        out.append({"name": name, "enabled": bool(item.get("enabled", False))})
    return out


def _enabled_observation_names(items: list[dict[str, Any]]) -> list[str]:
    return [str(item["name"]) for item in items if bool(item.get("enabled", False))]


def _history_steps_from_names(names: list[str], default: int = 1) -> int:
    max_frames = int(default)
    for name in names:
        m = _HISTORY_PATTERN.search(name)
        if m is None:
            continue
        try:
            frames = int(m.group(1))
        except ValueError:
            continue
        if frames > max_frames:
            max_frames = frames
    return max_frames


def _parse_encoder_modes(modes_raw: Any) -> tuple[list[dict[str, Any]], dict[str, int], dict[int, list[str]]]:
    if not isinstance(modes_raw, list):
        return [], {}, {}

    modes: list[dict[str, Any]] = []
    name_to_id: dict[str, int] = {}
    id_to_required: dict[int, list[str]] = {}

    for item in modes_raw:
        if not isinstance(item, dict):
            continue

        try:
            mode_id = int(item.get("mode_id"))
        except (TypeError, ValueError):
            continue

        mode_name = str(item.get("name", "")).strip()
        required_raw = item.get("required_observations", [])
        if not isinstance(required_raw, list):
            required_raw = []

        required: list[str] = []
        for value in required_raw:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                required.append(text)

        normalized = {
            "name": mode_name,
            "mode_id": mode_id,
            "required_observations": required,
        }
        modes.append(normalized)

        if mode_name:
            name_to_id[mode_name.lower()] = mode_id
        id_to_required[mode_id] = required

    return modes, name_to_id, id_to_required


def _parse_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def parse_observation_config(path: Path) -> Dict[str, Any]:
    file_exists = path.exists()
    raw: dict[str, Any] = {}

    if file_exists:
        loaded = yaml.safe_load(path.read_text()) or {}
        if isinstance(loaded, dict):
            raw = loaded
    has_official_sections = isinstance(raw.get("observations"), list) or isinstance(raw.get("encoder"), dict)

    observations_all = _normalize_observation_configs(raw.get("observations", []))
    used_default_observations = False

    # Official ParseFullConfig fallback behavior:
    # - missing file -> default observations
    # - official-format file with no valid observations -> default observations
    # Keep legacy-style files (history_steps/features only) untouched.
    if not file_exists or (has_official_sections and not observations_all):
        observations_all = _default_observation_configs()
        used_default_observations = True

    encoder_cfg = raw.get("encoder", {})
    if not isinstance(encoder_cfg, dict):
        encoder_cfg = {}

    encoder_dimension = _parse_int(encoder_cfg.get("dimension", 0), default=0)
    encoder_use_fp16 = bool(encoder_cfg.get("use_fp16", False))
    encoder_observations_all = _normalize_observation_configs(encoder_cfg.get("encoder_observations", []))
    encoder_modes, encoder_mode_name_to_id, encoder_mode_id_to_required = _parse_encoder_modes(
        encoder_cfg.get("encoder_modes", [])
    )

    has_token_state = any(item["name"] == "token_state" and bool(item.get("enabled", False)) for item in observations_all)

    parse_ok = True
    parse_error = ""

    # Match official token-state validation semantics.
    if has_token_state:
        if encoder_dimension <= 0:
            parse_ok = False
            parse_error = (
                "'token_state' observation is enabled but encoder section is missing or has invalid dimension"
            )
            observations_all = []
    else:
        # Match official behavior: encoder section is ignored when token_state is disabled.
        if encoder_dimension > 0:
            encoder_dimension = 0
            encoder_use_fp16 = False
            encoder_observations_all = []
            encoder_modes = []
            encoder_mode_name_to_id = {}
            encoder_mode_id_to_required = {}

    decoder_names = _enabled_observation_names(observations_all)
    encoder_names = _enabled_observation_names(encoder_observations_all)

    # Legacy fallback format still supported by tests/adapters.
    legacy_features = raw.get("features", [])
    if not isinstance(legacy_features, list):
        legacy_features = []

    explicit_history = raw.get("history_steps", None)
    if explicit_history is not None:
        history_steps = int(explicit_history)
    else:
        history_steps = _history_steps_from_names(
            decoder_names + encoder_names + [str(x) for x in legacy_features],
            default=1,
        )

    g1_mode_id = encoder_mode_name_to_id.get("g1")
    g1_required_observations = (
        encoder_mode_id_to_required.get(g1_mode_id, []) if g1_mode_id is not None else []
    )

    return {
        "history_steps": int(history_steps),
        "features": legacy_features,
        "observations": observations_all,
        "decoder_observations": decoder_names,
        "encoder_observations_all": encoder_observations_all,
        "encoder_observations": encoder_names,
        "encoder_dimension": int(encoder_dimension),
        "encoder_use_fp16": bool(encoder_use_fp16),
        "encoder_modes": encoder_modes,
        "encoder_mode_name_to_id": encoder_mode_name_to_id,
        "encoder_mode_id_to_required": encoder_mode_id_to_required,
        "g1_mode_id": g1_mode_id,
        "g1_required_observations": g1_required_observations,
        "parse_ok": bool(parse_ok),
        "parse_error": parse_error,
        "default_observations_used": bool(used_default_observations),
        "is_official_format": bool(has_official_sections or used_default_observations),
        "raw": raw,
    }
