from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.guidance.ode_guidance import FlowODEGuidance
from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator
from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    _ = yaml.safe_load(args.config.read_text())

    b, t = 4, 12
    context = torch.randn(b, 16)
    flow = FlowMatchingGenerator()
    wm = StructuredClosedLoopWorldModel()
    guidance = FlowODEGuidance(wm)

    unguided = flow.sample_unguided(b, t, context)

    anchor = flow.decode_reference(torch.zeros(b, t, flow.latent_dim))
    def grad_fn(x: torch.Tensor) -> torch.Tensor:
        return guidance.gradient(x, flow.decode_reference, anchor)

    guided = flow.sample_guided(b, t, context, grad_fn)
    print("eval_flow_guidance complete", {"unguided_mean": float(unguided.mean()), "guided_mean": float(guided.mean())})


if __name__ == "__main__":
    main()
