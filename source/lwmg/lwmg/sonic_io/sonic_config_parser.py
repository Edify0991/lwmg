from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import yaml


def parse_observation_config(path: Path) -> Dict[str, Any]:
    with path.open() as f:
        data = yaml.safe_load(f) or {}
    return {
        "history_steps": int(data.get("history_steps", 1)),
        "features": data.get("features", []),
        "raw": data,
    }
