from __future__ import annotations

from typing import Iterable


def success_rate(successes: Iterable[bool]) -> float:
    vals = list(successes)
    return float(sum(vals) / max(1, len(vals)))
