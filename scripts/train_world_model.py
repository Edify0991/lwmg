from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", type=str, choices=["nominal", "residual", "joint"], default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    stage = args.stage or cfg.get("train", {}).get("stage", "nominal")

    model = StructuredClosedLoopWorldModel()
    if stage == "residual":
        for p in model.nominal.parameters():
            p.requires_grad = False
        lr = 1e-3
    elif stage == "joint":
        lr = 1e-4
    else:
        lr = 1e-3

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for _ in range(3):
        b, t = 8, 10
        s0 = torch.randn(b, 32)
        refs = torch.randn(b, t, 29)
        ctrls = torch.randn(b, t, 29)
        hist = torch.randn(b, 4, 32)
        rollout = model.rollout(s0, refs, ctrls, hist)
        loss = rollout.pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"world model training complete, stage={stage}")


if __name__ == "__main__":
    main()
