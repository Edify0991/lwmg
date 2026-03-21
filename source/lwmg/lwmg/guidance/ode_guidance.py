from __future__ import annotations

from dataclasses import dataclass

import torch

from .cost_terms import (
    prior_anchor_cost,
    smoothness_cost,
    stability_cost,
    task_cost,
    tracking_feasibility_cost,
    uncertainty_cost,
)


@dataclass
class GuidanceWeights:
    task: float = 1.0
    tracking: float = 1.0
    stability: float = 1.0
    smoothness: float = 0.2
    uncertainty: float = 0.2
    prior: float = 0.1


class FlowODEGuidance:
    def __init__(self, world_model, weights: GuidanceWeights | None = None, lambda_guidance: float = 1.0, enabled: bool = True) -> None:
        self.world_model = world_model
        self.weights = weights or GuidanceWeights()
        self.lambda_guidance = lambda_guidance
        self.enabled = enabled

    def compute_guidance_cost(
        self,
        reference: torch.Tensor,
        unguided_reference: torch.Tensor,
        wm_rollout: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        c_task = task_cost(wm_rollout["task_progress"], wm_rollout["target_vel_error"])
        c_track = tracking_feasibility_cost(wm_rollout["tracking_error"])
        c_stab = stability_cost(
            wm_rollout["trunk_tilt"],
            wm_rollout["base_height"],
            wm_rollout["support_margin"],
            wm_rollout["slip"],
            wm_rollout["torque"],
        )
        c_smooth = smoothness_cost(reference)
        c_unc = uncertainty_cost(wm_rollout["uncertainty"])
        c_prior = prior_anchor_cost(reference, unguided_reference)
        return (
            self.weights.task * c_task
            + self.weights.tracking * c_track
            + self.weights.stability * c_stab
            + self.weights.smoothness * c_smooth
            + self.weights.uncertainty * c_unc
            + self.weights.prior * c_prior
        )

    def grad_guidance(self, latent_state: torch.Tensor, decode_fn, unguided_reference: torch.Tensor) -> torch.Tensor:
        if not self.enabled:
            return torch.zeros_like(latent_state)
        x = latent_state.detach().requires_grad_(True)
        ref = decode_fn(x)
        wm_rollout = self.world_model.rollout_from_reference(ref)
        cost = self.compute_guidance_cost(ref, unguided_reference, wm_rollout)
        grad = torch.autograd.grad(cost, x, retain_graph=False, create_graph=False)[0]
        return self.lambda_guidance * grad

    def guided_step(self, x_tau: torch.Tensor, tau: torch.Tensor, velocity_fn, decode_fn, unguided_reference: torch.Tensor) -> torch.Tensor:
        v = velocity_fn(x_tau, tau)
        g = self.grad_guidance(x_tau, decode_fn, unguided_reference)
        return v - g

    def guided_rollout(self, x0: torch.Tensor, velocity_fn, decode_fn, unguided_reference: torch.Tensor, n_steps: int = 16) -> torch.Tensor:
        x = x0
        dt = 1.0 / max(1, n_steps)
        for i in range(n_steps):
            tau = torch.full((x.shape[0], 1), i * dt, device=x.device, dtype=x.dtype)
            x = x + dt * self.guided_step(x, tau, velocity_fn, decode_fn, unguided_reference)
        return x
