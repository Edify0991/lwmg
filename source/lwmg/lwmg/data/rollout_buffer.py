from __future__ import annotations

from dataclasses import dataclass, field

from lwmg.envs.observations import Observation
from lwmg.envs.random_load_sampler import PrivilegedLoadLabel
from lwmg.envs.terminations import FailureFlags
from lwmg.references.reference_types import ReferenceTarget


@dataclass
class RolloutStep:
    observation: Observation
    privileged: PrivilegedLoadLabel
    reference: ReferenceTarget
    failure_flags: FailureFlags


@dataclass
class RolloutEpisode:
    steps: list[RolloutStep] = field(default_factory=list)
    split: str = "mixed"  # nominal | loaded | pair | mixed
    pair_id: str | None = None


@dataclass
class WMSequence:
    observations: list[Observation]
    hard_failure: list[bool]
    soft_failure: list[bool]
    split: str = "mixed"
