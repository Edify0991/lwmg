from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.world_model.losses import (
    nominal_rollout_loss,
    nominal_state_loss,
    paired_counterfactual_loss,
    residual_rollout_loss,
    residual_state_loss,
    residual_zero_loss,
)
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
        for p in model.residual.parameters():
            p.requires_grad = False
        for p in model.interaction.parameters():
            p.requires_grad = False
        lr = 1e-3

    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr)

    for _ in range(3):
        b, t = 8, 10
        s0 = torch.randn(b, 32)
        refs = torch.randn(b, t, 29)
        ctrls = torch.randn(b, t, 29)
        hist = torch.randn(b, 4, 32)
        rollout = model.rollout(s0, refs, ctrls, hist)

        loss = torch.zeros((), dtype=rollout.dtype)
        if stage in {"nominal", "joint"}:
            nom_pred = model.nominal(s0, refs[:, 0], ctrls[:, 0])
            loss = loss + nominal_state_loss(nom_pred, s0) + nominal_rollout_loss(rollout[:, 1:], rollout[:, 1:].detach())

        if stage in {"residual", "joint"}:
            s_nom = model.nominal(s0, refs[:, 0], ctrls[:, 0]).detach()
            z_int = model.encode_interaction(hist, refs[:, 0], s_nom)
            delta = model.residual(s0, refs[:, 0], ctrls[:, 0], z_int)
            loss = loss + residual_state_loss(delta, torch.zeros_like(delta))
            loss = loss + residual_rollout_loss(rollout[:, 1:], rollout[:, 1:].detach())
            loss = loss + residual_zero_loss(delta[: b // 2])
            loss = loss + paired_counterfactual_loss(rollout[: b // 2, -1], rollout[b // 2 :, -1], rollout[: b // 2, -1].detach(), rollout[b // 2 :, -1].detach())

        opt.zero_grad()
        loss.backward()
        opt.step()

    print(f"world model training complete, stage={stage}")


if __name__ == "__main__":
    main()
