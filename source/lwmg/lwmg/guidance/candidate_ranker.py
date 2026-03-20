from __future__ import annotations

from dataclasses import dataclass

import torch

from .costs import guidance_cost


@dataclass
class CandidateRanker:
    top_k: int = 8

    def rank(self, metrics: dict[str, torch.Tensor]) -> torch.Tensor:
        c = guidance_cost(**metrics)
        return torch.argsort(c)[: self.top_k]
