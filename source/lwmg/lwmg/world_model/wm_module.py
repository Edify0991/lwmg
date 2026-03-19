from __future__ import annotations

import torch
from torch import nn

from .context_encoder import ContextEncoder
from .prediction_heads import PredictionHeads
from .transition_model import TransitionModel


class LoadAwareWorldModel(nn.Module):
    def __init__(self, state_dim: int = 32, hidden_dim: int = 128, latent_dim: int = 16) -> None:
        super().__init__()
        self.context_encoder = ContextEncoder(state_dim, hidden_dim, latent_dim)
        self.transition = TransitionModel(state_dim, latent_dim, hidden_dim)
        self.heads = PredictionHeads(state_dim, hidden_dim)

    def forward(self, state: torch.Tensor, history: torch.Tensor) -> dict[str, torch.Tensor]:
        z_t = self.context_encoder(history)
        next_state = self.transition(state, z_t)
        out = self.heads(next_state)
        out["next_state"] = next_state
        out["z_t"] = z_t
        return out
