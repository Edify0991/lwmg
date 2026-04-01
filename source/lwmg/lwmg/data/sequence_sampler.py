from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class HistorySpec:
    """Window specification for slow/fast interaction histories."""

    h_slow_len: int = 8
    h_fast_len: int = 4


def sample_windows(length: int, window: int, stride: int = 1) -> list[tuple[int, int]]:
    if length <= 0 or window <= 0 or length < window:
        return []
    stride = max(1, int(stride))
    return [(i, i + window) for i in range(0, length - window + 1, stride)]


def filter_valid_windows(
    tags: Sequence[str],
    *,
    allow_post_impact: bool = False,
    corruption_tags: set[str] | None = None,
) -> list[int]:
    keep = {"stable", "near_failure", "pre_collapse"}
    if allow_post_impact:
        keep.add("post_impact")
    bad = corruption_tags or {"nan", "inf", "corrupted", "numeric_explosion"}
    return [i for i, tag in enumerate(tags) if tag in keep and tag not in bad]


def truncate_on_first_invalid(tags: Sequence[str], invalid_markers: set[str] | None = None) -> int:
    """Return valid prefix length before numerically corrupted tail begins."""

    bad = invalid_markers or {"post_impact", "nan", "inf", "corrupted", "numeric_explosion"}
    for i, tag in enumerate(tags):
        if tag in bad:
            return i
    return len(tags)


def extract_histories(sequence, step: int, spec: HistorySpec) -> tuple:
    """Extract h_slow / h_fast with zero-padding for insufficient early history."""

    if step < 0:
        raise ValueError("step must be non-negative")

    if len(sequence) == 0:
        raise ValueError("sequence must be non-empty")

    feat_dim = len(sequence[0])

    def _gather(length: int):
        start = max(0, step - length + 1)
        hist = sequence[start : step + 1]
        pad = length - len(hist)
        if pad <= 0:
            return hist
        return [[0.0] * feat_dim for _ in range(pad)] + list(hist)

    return _gather(spec.h_slow_len), _gather(spec.h_fast_len)
