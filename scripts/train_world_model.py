from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.world_model.wm_module import LoadAwareWorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    epochs = int(cfg["train"].get("epochs", 1))

    model = LoadAwareWorldModel(state_dim=32)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        state = torch.randn(8, 32)
        history = torch.randn(8, 4, 32)
        out = model(state, history)
        loss = (out["next_state"] - state).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    print("world model training complete")


if __name__ == "__main__":
    main()
