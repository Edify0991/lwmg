from __future__ import annotations

from typing import Sequence


def sample_windows(length: int, window: int) -> Sequence[tuple[int, int]]:
    return [(i, i + window) for i in range(0, max(0, length - window + 1))]
