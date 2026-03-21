from __future__ import annotations

from dataclasses import dataclass

import torch
from torch.utils.data import Dataset


@dataclass
class WorldModelSample:
    x: torch.Tensor
    y: torch.Tensor
    split: str
    pair_id: str | None


class WMDataset(Dataset[WorldModelSample]):
    def __init__(self, x: torch.Tensor, y: torch.Tensor, splits: list[str] | None = None, pair_ids: list[str | None] | None = None, mode: str = "mixed") -> None:
        self.x = x
        self.y = y
        self.splits = splits or ["mixed"] * x.shape[0]
        self.pair_ids = pair_ids or [None] * x.shape[0]
        self.mode = mode
        self.indices = [i for i, s in enumerate(self.splits) if mode == "mixed" or s == mode]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> WorldModelSample:
        i = self.indices[idx]
        return WorldModelSample(x=self.x[i], y=self.y[i], split=self.splits[i], pair_id=self.pair_ids[i])
