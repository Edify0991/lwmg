from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from lwmg.envs.observations import Observation
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
