from __future__ import annotations

from typing import Iterable


def failure_rate(flags: Iterable[bool]) -> float:
    vals = list(flags)
    return float(sum(vals) / max(1, len(vals)))
