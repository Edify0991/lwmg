from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PayloadCollectionConfig:
    mode: str = "mixed"  # wrench | rigid_payload | mixed
    wrench_probability: float = 0.5
    rigid_payload_probability: float = 0.5


def choose_payload_mode(step: int, cfg: PayloadCollectionConfig) -> str:
    """Deterministic lightweight mode selector for rollout collection scripts."""

    if cfg.mode in {"wrench", "rigid_payload"}:
        return cfg.mode
    return "wrench" if step % 2 == 0 else "rigid_payload"
