from __future__ import annotations

import argparse
from pathlib import Path

import torch
import yaml

from lwmg.guidance.ode_guidance import FlowODEGuidance
from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator
from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def _metrics(ref: torch.Tensor, wm: StructuredClosedLoopWorldModel) -> dict[str, float]:
    roll = wm.rollout_from_reference(ref)
    return {
        "task_progress": float(roll["task_progress"].mean()),
        "tracking_error": float(roll["tracking_error"].mean()),
        "stability_tilt": float(roll["trunk_tilt"].mean()),
        "torque_proxy": float(roll["torque"].mean()),
        "uncertainty": float(roll["uncertainty"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    _ = yaml.safe_load(args.config.read_text())

    b, t = 4, 12
    flow = FlowMatchingGenerator()
    wm = StructuredClosedLoopWorldModel()
    guidance = FlowODEGuidance(wm)

    for load_scale in [0.0, 0.5, 1.0]:
        context = torch.randn(b, 16) * (1.0 + load_scale)
        unguided = flow.sample_unguided(b, t, context)
        anchor = unguided.detach()

        def grad_fn(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            return guidance.grad_guidance(x, flow.decode_reference, anchor)

        guided = flow.sample_guided(b, t, context, grad_fn)
        print(
            f"load_scale={load_scale}",
            {"unguided": _metrics(unguided, wm), "guided": _metrics(guided, wm)},
        )


if __name__ == "__main__":
    main()
