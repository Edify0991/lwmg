from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class WorldModelSample:
    x: torch.Tensor
    y: torch.Tensor


class WMDataset(Dataset[WorldModelSample]):
    def __init__(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.x = x
        self.y = y

    def __len__(self) -> int:
        return self.x.shape[0]

    def __getitem__(self, idx: int) -> WorldModelSample:
        return WorldModelSample(x=self.x[idx], y=self.y[idx])
