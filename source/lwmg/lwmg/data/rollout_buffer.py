from __future__ import annotations

from dataclasses import dataclass, field

from lwmg.envs.observations import Observation
from lwmg.envs.random_load_sampler import PrivilegedLoadLabel
from lwmg.envs.terminations import FailureFlags
from lwmg.references.reference_types import ReferenceTarget


@dataclass
class PayloadMetadata:
    payload_mass: float | None = None
    payload_com_shift: tuple[float, float, float] | None = None
    payload_inertia: tuple[float, float, float] | None = None
    load_regime: str = "none"  # wrench | rigid_payload | mixed | none
    pair_id: str | None = None


@dataclass
class RolloutStep:
    observation: Observation
    privileged: PrivilegedLoadLabel
    reference: ReferenceTarget
    failure_flags: FailureFlags
    action: list[float] | None = None


@dataclass
class RolloutEpisode:
    steps: list[RolloutStep] = field(default_factory=list)
    split: str = "mixed"  # nominal | loaded | pair | mixed
    pair_id: str | None = None
    payload: PayloadMetadata = field(default_factory=PayloadMetadata)


@dataclass
class WMSequence:
    observations: list[Observation]
    hard_failure: list[bool]
    soft_failure: list[bool]
    split: str = "mixed"
    pair_id: str | None = None
    payload: PayloadMetadata = field(default_factory=PayloadMetadata)


def episode_to_wm_sequence(episode: RolloutEpisode) -> WMSequence:
    """Convert buffered rollout episode into a world-model sequence container."""

    observations = [step.observation for step in episode.steps]
    hard_failure = [step.failure_flags.hard_fail for step in episode.steps]
    soft_failure = [step.failure_flags.soft_fail for step in episode.steps]
    payload = episode.payload
    if payload.pair_id is None:
        payload = PayloadMetadata(
            payload_mass=payload.payload_mass,
            payload_com_shift=payload.payload_com_shift,
            payload_inertia=payload.payload_inertia,
            load_regime=payload.load_regime,
            pair_id=episode.pair_id,
        )

    return WMSequence(
        observations=observations,
        hard_failure=hard_failure,
        soft_failure=soft_failure,
        split=episode.split,
        pair_id=episode.pair_id,
        payload=payload,
    )
