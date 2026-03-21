from __future__ import annotations

import torch
from torch import nn

from .base_flow_generator import BaseFlowGenerator
from .flow_decoder import FlowDecoder
from .flow_objectives import flow_objective
from .ode_solver import integrate_ode


class FlowMatchingGenerator(nn.Module, BaseFlowGenerator):
    def __init__(self, latent_dim: int = 32, context_dim: int = 16, ref_dim: int = 29) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.context_proj = nn.Linear(context_dim + 1, latent_dim)
        self.decoder = FlowDecoder(latent_dim, ref_dim)

    def velocity_field(self, x: torch.Tensor, tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        ctx = context.unsqueeze(1).expand(-1, x.shape[1], -1)
        t = tau.unsqueeze(1).expand(-1, x.shape[1], -1)
        return torch.tanh(self.context_proj(torch.cat([ctx, t], dim=-1))) - 0.1 * x

    def decode_reference(self, latent_traj: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent_traj)

    def sample_unguided(self, batch_size: int, horizon: int, context: torch.Tensor) -> torch.Tensor:
        x0 = torch.randn(batch_size, horizon, self.latent_dim, device=context.device)
        ode_fn = lambda tau, x: self.velocity_field(x, tau, context)
        lat = integrate_ode(ode_fn, x0)
        return self.decode_reference(lat)

    def sample_guided(self, batch_size: int, horizon: int, context: torch.Tensor, guidance_fn) -> torch.Tensor:
        x0 = torch.randn(batch_size, horizon, self.latent_dim, device=context.device)

        def ode_fn(tau, x):
            v = self.velocity_field(x, tau, context)
            grad = guidance_fn(x, tau)
            return v - grad

        lat = integrate_ode(ode_fn, x0)
        return self.decode_reference(lat)

    def training_loss(self, x: torch.Tensor, tau: torch.Tensor, context: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
        pred = self.velocity_field(x, tau, context)
        return flow_objective(pred, target_velocity, family="flow_matching")
