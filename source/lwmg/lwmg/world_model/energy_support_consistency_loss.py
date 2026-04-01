from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn


@dataclass(frozen=True)
class EnergySupportConfig:
    """Hyper-parameters for proxy energy/support consistency losses."""

    gravity: float = 9.81
    root_height_index: int = 0
    posture_slice: tuple[int, int] = (1, 13)
    velocity_slice: tuple[int, int] = (13, 25)
    posture_potential_weight: float = 0.05
    action_work_coeff: float = 0.05
    wrench_work_coeff: float = 0.01
    dissipation_coeff: float = 0.02
    support_margin_safe: float = 0.015
    support_beta: float = 10.0
    stance_slip_max: float = 0.10
    slow_latent_margin: float = 0.0


class EnergySupportConsistencyLoss(nn.Module):
    """Practical mechanics-aware regularizer for closed-loop world model training."""

    def __init__(self, config: EnergySupportConfig | None = None) -> None:
        super().__init__()
        self.cfg = config or EnergySupportConfig()

    def _split_state(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        root_h = state[..., self.cfg.root_height_index]
        p0, p1 = self.cfg.posture_slice
        v0, v1 = self.cfg.velocity_slice
        posture = state[..., p0:p1]
        velocity = state[..., v0:v1]
        return root_h, posture, velocity

    def mechanical_energy_proxy(self, state: torch.Tensor) -> torch.Tensor:
        root_h, posture, velocity = self._split_state(state)
        kinetic = 0.5 * (velocity * velocity).sum(dim=-1)
        potential = self.cfg.gravity * root_h + self.cfg.posture_potential_weight * (posture * posture).sum(dim=-1)
        return kinetic + potential

    def energy_consistency_loss(
        self,
        state_t: torch.Tensor,
        state_tp1: torch.Tensor,
        action: torch.Tensor,
        external_wrench: torch.Tensor | None = None,
    ) -> torch.Tensor:
        e_t = self.mechanical_energy_proxy(state_t)
        e_tp1 = self.mechanical_energy_proxy(state_tp1)
        _, _, velocity_t = self._split_state(state_t)

        act_work = self.cfg.action_work_coeff * (action * action).sum(dim=-1)
        if external_wrench is None:
            int_work = torch.zeros_like(act_work)
        else:
            int_work = self.cfg.wrench_work_coeff * torch.abs(external_wrench).sum(dim=-1)
        dissipation = self.cfg.dissipation_coeff * (velocity_t * velocity_t).sum(dim=-1)

        residual = (e_tp1 - e_t) - (act_work + int_work - dissipation)
        return F.huber_loss(residual, torch.zeros_like(residual), reduction="mean")

    def support_consistency_loss(
        self,
        support_margin: torch.Tensor | None,
        stance_foot_slip: torch.Tensor | None,
    ) -> torch.Tensor:
        if support_margin is None and stance_foot_slip is None:
            return torch.zeros(())
        ref = support_margin if support_margin is not None else stance_foot_slip
        assert ref is not None
        loss = torch.zeros((), device=ref.device, dtype=ref.dtype)

        if support_margin is not None:
            margin_penalty = F.softplus((self.cfg.support_margin_safe - support_margin) * self.cfg.support_beta)
            loss = loss + margin_penalty.mean()

        if stance_foot_slip is not None:
            slip_mag = torch.abs(stance_foot_slip)
            slip_over = F.relu(slip_mag - self.cfg.stance_slip_max)
            loss = loss + F.huber_loss(slip_over, torch.zeros_like(slip_over), reduction="mean")

        return loss

    def slow_latent_alignment_loss(
        self,
        z_slow: torch.Tensor,
        load_regime: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if load_regime is None or z_slow.shape[0] <= 1:
            return torch.zeros((), device=z_slow.device, dtype=z_slow.dtype)

        # Same-regime pairs are encouraged to be close despite different references.
        z1 = z_slow.unsqueeze(1)
        z2 = z_slow.unsqueeze(0)
        dist = torch.norm(z1 - z2, dim=-1)

        regime_eq = load_regime.unsqueeze(1) == load_regime.unsqueeze(0)
        mask = regime_eq & ~torch.eye(z_slow.shape[0], device=z_slow.device, dtype=torch.bool)
        if not mask.any():
            return torch.zeros((), device=z_slow.device, dtype=z_slow.dtype)

        selected = dist[mask]
        target = torch.full_like(selected, self.cfg.slow_latent_margin)
        return F.huber_loss(selected, target, reduction="mean")

    def forward(
        self,
        state_t: torch.Tensor,
        state_tp1: torch.Tensor,
        action: torch.Tensor,
        *,
        external_wrench: torch.Tensor | None = None,
        support_margin: torch.Tensor | None = None,
        stance_foot_slip: torch.Tensor | None = None,
        z_slow: torch.Tensor | None = None,
        load_regime: torch.Tensor | None = None,
        energy_weight: float = 1.0,
        support_weight: float = 1.0,
        slow_latent_weight: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        energy = self.energy_consistency_loss(state_t, state_tp1, action, external_wrench=external_wrench)
        support = self.support_consistency_loss(support_margin, stance_foot_slip)
        if z_slow is None:
            slow_align = torch.zeros((), device=energy.device, dtype=energy.dtype)
        else:
            slow_align = self.slow_latent_alignment_loss(z_slow, load_regime=load_regime)

        total = energy_weight * energy + support_weight * support + slow_latent_weight * slow_align
        return {
            "energy_consistency": energy,
            "support_consistency": support,
            "slow_latent_alignment": slow_align,
            "total": total,
        }
