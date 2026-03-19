import torch

from lwmg.envs.observations import Observation
from lwmg.trackers.sonic_obs_builder import SonicObservationBuilder
from lwmg.trackers.sonic_obs_mapping import SonicObservationMapping


def test_sonic_obs_builder_vector() -> None:
    obs = Observation(
        q=torch.zeros(12),
        dq=torch.zeros(12),
        imu_accel=torch.zeros(3),
        imu_gyro=torch.zeros(3),
        prev_action=torch.zeros(12),
        contacts=torch.ones(2),
        tracking_error_summary=torch.zeros(1),
    )
    vec = SonicObservationBuilder(SonicObservationMapping.default()).build(obs)
    assert vec.ndim == 1
    assert vec.shape[0] > 12
