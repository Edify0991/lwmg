from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator
from lwmg.references.flows.mean_flow_generator import MeanFlowGenerator
from lwmg.references.flows.rectified_flow_generator import RectifiedFlowGenerator


def _make_generator(flow_family: str):
    if flow_family == "flow_matching":
        return FlowMatchingGenerator()
    if flow_family == "rectified_flow":
        return RectifiedFlowGenerator()
    if flow_family == "mean_flow":
        return MeanFlowGenerator()
    raise ValueError(f"Unsupported flow_family={flow_family}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train = cfg.get("train", {})
    flow_family = train.get("flow_family", "flow_matching")

    model = _make_generator(flow_family)
    opt = torch.optim.Adam(model.parameters(), lr=float(train.get("lr", 1e-3)))
    b, t = int(train.get("batch_size", 32)), int(train.get("horizon", 20))

    for _ in range(3):
        x = torch.randn(b, t, model.latent_dim)
        tau = torch.rand(b, 1)
        context = torch.randn(b, 16)
        target = torch.zeros_like(x)
        loss = model.training_loss(x, tau, context, target)
        opt.zero_grad()
        loss.backward()
        opt.step()
    print(f"flow training complete, flow_family={flow_family}")


if __name__ == "__main__":
    main()
