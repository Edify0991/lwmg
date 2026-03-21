from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator
from lwmg.references.flows.flow_objectives import flow_objective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    family = cfg.get("train", {}).get("family", "flow_matching")

    model = FlowMatchingGenerator()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(3):
        b, t = 8, 10
        x = torch.randn(b, t, model.latent_dim)
        tau = torch.rand(b, 1)
        context = torch.randn(b, 16)
        pred = model.velocity_field(x, tau, context)
        target = torch.zeros_like(pred)
        loss = flow_objective(pred, target, family=family)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"flow training complete, family={family}")


if __name__ == "__main__":
    main()
