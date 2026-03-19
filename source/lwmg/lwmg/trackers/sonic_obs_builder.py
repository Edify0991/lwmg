from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from lwmg.envs.observations import Observation
from .sonic_obs_mapping import SonicObservationMapping


@dataclass
class SonicObservationBuilder:
    mapping: SonicObservationMapping
    history_steps: int = 1

    def build(self, current: Observation, history: Iterable[Observation] | None = None) -> torch.Tensor:
        obs_chunks = [
            current.q,
            current.dq,
            current.imu_accel,
            current.imu_gyro,
            current.prev_action,
            current.contacts,
            current.tracking_error_summary,
        ]
        if history:
            for item in history:
                obs_chunks.extend([item.q, item.dq])
        return torch.cat([x.flatten() for x in obs_chunks], dim=0)
