from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseFlowGenerator(ABC):
    """Unified flow-family interface for reference generation."""

    @abstractmethod
    def velocity_field(self, x: torch.Tensor, tau: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def decode_reference(self, latent_traj: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_unguided(self, batch_size: int, horizon: int, context: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def sample_guided(self, batch_size: int, horizon: int, context: torch.Tensor, guidance_fn) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def training_loss(self, x: torch.Tensor, tau: torch.Tensor, context: torch.Tensor, target_velocity: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
