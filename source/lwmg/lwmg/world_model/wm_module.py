from __future__ import annotations

import torch
from torch import nn

from .interaction_encoder import InteractionEncoder
from .nominal_transition_model import NominalTransitionModel
from .residual_transition_model import ResidualTransitionModel


class StructuredClosedLoopWorldModel(nn.Module):
    def __init__(self, state_dim: int = 32, ref_dim: int = 29, ctrl_dim: int = 29, latent_dim: int = 32, ensemble: list[nn.Module] | None = None) -> None:
        super().__init__()
        self.nominal = NominalTransitionModel(state_dim, ref_dim, ctrl_dim)
        self.interaction = InteractionEncoder(state_dim, ref_dim, latent_dim=latent_dim)
        self.residual = ResidualTransitionModel(state_dim, ref_dim, ctrl_dim, latent_dim=latent_dim)
        self.ensemble = ensemble or []

    def encode_interaction(self, history: torch.Tensor, r_t: torch.Tensor, s_nom_tp1: torch.Tensor) -> torch.Tensor:
        return self.interaction(history, r_t, s_nom_tp1)

    def predict_step(self, s_t: torch.Tensor, r_t: torch.Tensor, u_t: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        s_nom = self.nominal(s_t, r_t, u_t)
        z_int = self.encode_interaction(history, r_t, s_nom)
        delta = self.residual(s_t, r_t, u_t, z_int)
        return s_nom + delta

    def step(self, s_t: torch.Tensor, r_t: torch.Tensor, u_t: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        return self.predict_step(s_t, r_t, u_t, history)

    def rollout(self, s0: torch.Tensor, references: torch.Tensor, controls: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        states = [s0]
        s_t = s0
        for t in range(references.shape[1]):
            s_t = self.predict_step(s_t, references[:, t], controls[:, t], history)
            states.append(s_t)
        return torch.stack(states, dim=1)

    def predict_uncertainty(self, s0: torch.Tensor, references: torch.Tensor, controls: torch.Tensor, history: torch.Tensor) -> torch.Tensor:
        if not self.ensemble:
            pred = self.rollout(s0, references, controls, history)
            return torch.zeros_like(pred)
        preds = torch.stack([m.rollout(s0, references, controls, history) for m in self.ensemble], dim=0)
        return preds.var(dim=0, unbiased=False)

    def rollout_from_reference(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        trunk_tilt = torch.abs(reference[..., 0])
        base_height = 0.8 - 0.1 * torch.abs(reference[..., 1])
        tracking_error = torch.abs(reference).mean(dim=-1)
        support_margin = 0.2 - 0.05 * torch.abs(reference[..., 2])
        slip = torch.abs(reference[:, 1:, :2] - reference[:, :-1, :2]).mean(dim=-1)
        torque = torch.abs(reference[..., :6])
        return {
            "task_progress": torch.sigmoid(reference[..., 0].mean(dim=1)),
            "target_vel_error": torch.abs(reference[..., 1].mean(dim=1)),
            "tracking_error": tracking_error,
            "trunk_tilt": trunk_tilt,
            "base_height": base_height,
            "support_margin": support_margin,
            "slip": slip,
            "torque": torque,
            "uncertainty": reference.var(dim=-1, unbiased=False),
        }
