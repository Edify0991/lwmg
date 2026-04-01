from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lwmg.envs.observations import Observation
from lwmg.sonic_io.sonic_config_parser import parse_observation_config
from lwmg.trackers.sonic_obs_builder import SonicObservationBuilder


def _obs() -> Observation:
    return Observation(
        q=torch.zeros(12),
        dq=torch.zeros(12),
        imu_accel=torch.zeros(3),
        imu_gyro=torch.zeros(3),
        prev_action=torch.zeros(12),
        contacts=torch.ones(2),
        tracking_error_summary=torch.zeros(1),
    )


def test_sonic_obs_builder_history_1(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("mapping:\n  q: joint_position\n  dq: joint_velocity\n")
    obs_cfg = tmp_path / "observation_config.yaml"
    obs_cfg.write_text("history_steps: 1\n")

    vec = SonicObservationBuilder(mapping_path=mapping, observation_config_path=obs_cfg, debug=True).build(_obs())
    assert vec.ndim == 1
    assert vec.shape[0] == 24


def test_sonic_obs_builder_history_4_requires_frames(tmp_path: Path) -> None:
    mapping = tmp_path / "mapping.yaml"
    mapping.write_text("mapping:\n  q: joint_position\n")
    obs_cfg = tmp_path / "observation_config.yaml"
    obs_cfg.write_text("history_steps: 4\n")

    builder = SonicObservationBuilder(mapping_path=mapping, observation_config_path=obs_cfg)
    with pytest.raises(ValueError):
        builder.build(_obs(), history=[])

    vec = builder.build(_obs(), history=[_obs(), _obs(), _obs()])
    assert vec.shape[0] == 48


def test_parse_observation_config_official_style(tmp_path: Path) -> None:
    obs_cfg = tmp_path / "observation_config.yaml"
    obs_cfg.write_text(
        """
observations:
  - name: token_state
    enabled: true
  - name: his_body_joint_positions_10frame_step1
    enabled: true
  - name: his_last_actions_10frame_step1
    enabled: false
encoder:
  dimension: 64
  encoder_observations:
    - name: encoder_mode_4
      enabled: true
    - name: motion_joint_positions_10frame_step5
      enabled: true
  encoder_modes:
    - name: g1
      mode_id: 0
      required_observations: [encoder_mode_4]
""".strip()
        + "\n"
    )

    parsed = parse_observation_config(obs_cfg)
    assert parsed["parse_ok"] is True
    assert parsed["is_official_format"] is True
    assert parsed["history_steps"] == 10
    assert parsed["encoder_dimension"] == 64
    assert parsed["decoder_observations"] == [
        "token_state",
        "his_body_joint_positions_10frame_step1",
    ]
    assert parsed["encoder_observations"] == [
        "encoder_mode_4",
        "motion_joint_positions_10frame_step5",
    ]
    assert isinstance(parsed["encoder_modes"], list)


def test_parse_observation_config_legacy_style(tmp_path: Path) -> None:
    obs_cfg = tmp_path / "observation_config.yaml"
    obs_cfg.write_text("history_steps: 4\nfeatures:\n  - q\n  - dq\n")

    parsed = parse_observation_config(obs_cfg)
    assert parsed["parse_ok"] is True
    assert parsed["is_official_format"] is False
    assert parsed["history_steps"] == 4
    assert parsed["features"] == ["q", "dq"]
    assert parsed["decoder_observations"] == []
    assert parsed["encoder_observations"] == []


def test_parse_observation_config_token_state_requires_encoder(tmp_path: Path) -> None:
    obs_cfg = tmp_path / "observation_config.yaml"
    obs_cfg.write_text(
        """
observations:
  - name: token_state
    enabled: true
""".strip()
        + "\n"
    )

    parsed = parse_observation_config(obs_cfg)
    assert parsed["parse_ok"] is False
    assert "token_state" in parsed["parse_error"]
    assert parsed["decoder_observations"] == []


def test_parse_observation_config_encoder_ignored_without_token_state(tmp_path: Path) -> None:
    obs_cfg = tmp_path / "observation_config.yaml"
    obs_cfg.write_text(
        """
observations:
  - name: his_body_joint_positions_10frame_step1
    enabled: true
encoder:
  dimension: 64
  use_fp16: true
  encoder_observations:
    - name: encoder_mode_4
      enabled: true
""".strip()
        + "\n"
    )

    parsed = parse_observation_config(obs_cfg)
    assert parsed["parse_ok"] is True
    assert parsed["encoder_dimension"] == 0
    assert parsed["encoder_use_fp16"] is False
    assert parsed["encoder_observations"] == []


def test_parse_observation_config_missing_file_uses_default_observations(tmp_path: Path) -> None:
    obs_cfg = tmp_path / "missing_observation_config.yaml"

    parsed = parse_observation_config(obs_cfg)
    assert parsed["parse_ok"] is True
    assert parsed["default_observations_used"] is True
    assert parsed["is_official_format"] is True
    assert parsed["decoder_observations"] == [
        "motion_joint_positions",
        "motion_joint_velocities",
        "motion_anchor_orientation",
        "base_angular_velocity",
        "body_joint_positions",
        "body_joint_velocities",
        "last_actions",
    ]

