from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


@dataclass
class SonicObservationMapping:
    mapping: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def default(cls) -> "SonicObservationMapping":
        return cls(
            mapping={
                "q": "joint_position",
                "dq": "joint_velocity",
                "imu_accel": "imu_accel",
                "imu_gyro": "imu_gyro",
                "prev_action": "prev_action",
                "contacts": "foot_contacts",
                "tracking_error_summary": "tracking_error",
            }
        )
