from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.sonic_io.sonic_reference_exporter import SonicReferenceExporter


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    out_dir = Path(cfg["export"]["out_dir"])

    t = 100
    data = {
        "joint_pos": torch.zeros(t, 12),
        "joint_vel": torch.zeros(t, 12),
        "body_pos": torch.zeros(t, 3),
        "body_quat": torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(t, 1),
    }
    SonicReferenceExporter().export(out_dir, data)
    print(f"exported to {out_dir}")


if __name__ == "__main__":
    main()
