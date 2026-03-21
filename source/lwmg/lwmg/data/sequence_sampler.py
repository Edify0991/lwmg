from __future__ import annotations

from typing import Sequence


def sample_windows(length: int, window: int) -> Sequence[tuple[int, int]]:
    return [(i, i + window) for i in range(0, max(0, length - window + 1))]


def filter_valid_windows(tags: Sequence[str]) -> list[int]:
    keep = {"stable", "near_failure", "pre_collapse"}
    return [i for i, t in enumerate(tags) if t in keep]
