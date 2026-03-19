from __future__ import annotations

from abc import ABC, abstractmethod

from .reference_types import ReferenceTarget


class BaseReferenceGenerator(ABC):
    @abstractmethod
    def generate(self, batch_size: int) -> ReferenceTarget:
        raise NotImplementedError
