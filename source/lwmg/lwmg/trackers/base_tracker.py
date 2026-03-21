from __future__ import annotations

from abc import ABC, abstractmethod

import torch


class BaseTracker(ABC):
    @abstractmethod
    def act(self, observation: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError
