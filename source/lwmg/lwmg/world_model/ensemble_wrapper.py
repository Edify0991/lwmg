from __future__ import annotations

import torch


class EnsembleWrapper:
    def __init__(self, models: list) -> None:
        self.models = models

    def rollout(self, *args, **kwargs) -> tuple[torch.Tensor, torch.Tensor]:
        preds = [m.rollout(*args, **kwargs) for m in self.models]
        stack = torch.stack(preds, dim=0)
        mean = stack.mean(dim=0)
        var = stack.var(dim=0, unbiased=False)
        return mean, var
