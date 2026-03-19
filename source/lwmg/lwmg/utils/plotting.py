from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt


def save_curve(values: Iterable[float], out_path: Path, title: str) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(list(values))
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
