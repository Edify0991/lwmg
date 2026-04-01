from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass(frozen=True)
class LatentReferenceOptimizerConfig:
    num_steps: int = 16
    lr: float = 5.0e-2
    weight_decay: float = 0.0
    grad_clip_norm: float = 5.0
    latent_clip: float = 3.0


class LatentReferenceOptimizer:
    """Test-time optimizer over deformation latent z_def.

    Default stable path: optimize low-dimensional latent deformation instead of
    unconstrained full joint-space trajectory variables.
    """

    def __init__(self, deformation_decoder, config: LatentReferenceOptimizerConfig | None = None) -> None:
        self.deformation_decoder = deformation_decoder
        self.cfg = config or LatentReferenceOptimizerConfig()

    def initialize_latent(self, batch_size: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(batch_size, self.deformation_decoder.cfg.latent_dim, device=device, dtype=dtype)

    def decode_reference(
        self,
        nominal_reference: torch.Tensor,
        z_def: torch.Tensor,
        active_groups: list[str] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        delta_r, r_star = self.deformation_decoder(
            nominal_reference,
            z_def,
            active_groups=active_groups,
            return_delta=True,
        )
        return delta_r, r_star

    def compute_rollout_cost(
        self,
        *,
        world_model,
        nominal_reference: torch.Tensor,
        z_def: torch.Tensor,
        current_state: torch.Tensor,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
        controls: torch.Tensor | None = None,
        active_groups: list[str] | None = None,
        task: str = "balance",
        goal: dict[str, Any] | None = None,
        cost_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        delta_r, r_star = self.decode_reference(
            nominal_reference=nominal_reference,
            z_def=z_def,
            active_groups=active_groups,
        )

        if controls is None:
            controls = torch.zeros(
                r_star.shape[0],
                r_star.shape[1],
                world_model.ctrl_dim,
                device=r_star.device,
                dtype=r_star.dtype,
            )

        rollout = world_model.rollout(current_state, r_star, controls, h_slow, h_fast)
        score = world_model.score_reference(
            reference=r_star,
            rollout_states=rollout,
            nominal_reference=nominal_reference,
            delta_reference=delta_r,
            task=task,
            goal=goal,
            weights=cost_weights,
        )
        return score["total"], score

    def optimize(
        self,
        *,
        world_model,
        nominal_reference: torch.Tensor,
        current_state: torch.Tensor,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
        controls: torch.Tensor | None = None,
        active_groups: list[str] | None = None,
        task: str = "balance",
        goal: dict[str, Any] | None = None,
        cost_weights: dict[str, float] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        # Test-time adaptation optimizes only z_def; keep other inputs as constants.
        nominal_reference = nominal_reference.detach()
        current_state = current_state.detach()
        h_slow = h_slow.detach()
        if h_fast is not None:
            h_fast = h_fast.detach()
        if controls is not None:
            controls = controls.detach()

        z_def = self.initialize_latent(
            batch_size=nominal_reference.shape[0],
            device=nominal_reference.device,
            dtype=nominal_reference.dtype,
        )
        z_def = z_def.requires_grad_(True)

        opt = torch.optim.Adam([z_def], lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)
        best_score: dict[str, torch.Tensor] | None = None
        best_cost: float | None = None

        for _ in range(self.cfg.num_steps):
            total_cost, score = self.compute_rollout_cost(
                world_model=world_model,
                nominal_reference=nominal_reference,
                z_def=z_def,
                current_state=current_state,
                h_slow=h_slow,
                h_fast=h_fast,
                controls=controls,
                active_groups=active_groups,
                task=task,
                goal=goal,
                cost_weights=cost_weights,
            )

            opt.zero_grad()
            total_cost.backward()
            torch.nn.utils.clip_grad_norm_([z_def], self.cfg.grad_clip_norm)
            opt.step()

            with torch.no_grad():
                z_def.clamp_(-self.cfg.latent_clip, self.cfg.latent_clip)
                cur = float(total_cost.detach().item())
                if best_cost is None or cur < best_cost:
                    best_cost = cur
                    best_score = {k: v.detach().clone() for k, v in score.items()}

        _, r_star = self.decode_reference(
            nominal_reference=nominal_reference,
            z_def=z_def.detach(),
            active_groups=active_groups,
        )

        assert best_score is not None
        return z_def.detach(), r_star.detach(), best_score
