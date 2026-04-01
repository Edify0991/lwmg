from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import torch
from torch.utils.data import Dataset


@dataclass
class WorldModelSample:
    """Single world-model training sample.

    Legacy fields (`x`, `y`) are kept for compatibility with earlier tests/scripts.
    Structured fields (`s_t`, `s_tp1`, `r_t`, `u_t`, `h_slow`, `h_fast`) support the
    endpoint-load-aware residual training pipeline.
    """

    x: torch.Tensor
    y: torch.Tensor
    split: str = "mixed"
    pair_id: str | None = None

    s_t: torch.Tensor | None = None
    s_tp1: torch.Tensor | None = None
    r_t: torch.Tensor | None = None
    u_t: torch.Tensor | None = None
    h_slow: torch.Tensor | None = None
    h_fast: torch.Tensor | None = None

    payload_mass: torch.Tensor | None = None
    payload_com_shift: torch.Tensor | None = None
    payload_inertia: torch.Tensor | None = None
    load_regime: str | None = None


def _split_match(mode: str, split: str) -> bool:
    if mode == "mixed":
        return True
    if mode in {"load", "loaded"}:
        return split in {"load", "loaded"}
    return split == mode


def _to_world_model_sample(obj: WorldModelSample | dict[str, Any]) -> WorldModelSample:
    if isinstance(obj, WorldModelSample):
        return obj

    x = obj.get("x")
    y = obj.get("y")
    s_t = obj.get("s_t")
    s_tp1 = obj.get("s_tp1")
    if x is None:
        x = s_t
    if y is None:
        y = s_tp1
    if x is None or y is None:
        raise ValueError("Each structured sample must provide (`x`,`y`) or (`s_t`,`s_tp1`).")

    return WorldModelSample(
        x=x,
        y=y,
        split=str(obj.get("split", "mixed")),
        pair_id=obj.get("pair_id"),
        s_t=s_t,
        s_tp1=s_tp1,
        r_t=obj.get("r_t"),
        u_t=obj.get("u_t"),
        h_slow=obj.get("h_slow"),
        h_fast=obj.get("h_fast"),
        payload_mass=obj.get("payload_mass"),
        payload_com_shift=obj.get("payload_com_shift"),
        payload_inertia=obj.get("payload_inertia"),
        load_regime=obj.get("load_regime"),
    )


class WMDataset(Dataset[WorldModelSample]):
    """Dataset that supports D_nom / D_load / D_pair subsets and legacy usage."""

    def __init__(
        self,
        x: torch.Tensor | None = None,
        y: torch.Tensor | None = None,
        *,
        samples: Sequence[WorldModelSample | dict[str, Any]] | None = None,
        splits: list[str] | None = None,
        pair_ids: list[str | None] | None = None,
        mode: str = "mixed",
    ) -> None:
        self.mode = mode

        if samples is not None:
            self.samples = [_to_world_model_sample(s) for s in samples]
        else:
            if x is None or y is None:
                raise ValueError("WMDataset requires either (`samples`) or both (`x`, `y`).")
            use_splits = splits or ["mixed"] * x.shape[0]
            use_pair_ids = pair_ids or [None] * x.shape[0]
            self.samples = [
                WorldModelSample(
                    x=x[i],
                    y=y[i],
                    split=use_splits[i],
                    pair_id=use_pair_ids[i],
                    s_t=x[i],
                    s_tp1=y[i],
                )
                for i in range(x.shape[0])
            ]

        self.indices = [
            i
            for i, sample in enumerate(self.samples)
            if _split_match(self.mode, sample.split)
            and not (self.mode == "pair" and sample.pair_id is None)
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> WorldModelSample:
        return self.samples[self.indices[idx]]

    @classmethod
    def from_tensor_dict(cls, tensors: dict[str, Any], mode: str = "mixed") -> "WMDataset":
        if "samples" in tensors:
            return cls(samples=tensors["samples"], mode=mode)

        x = tensors.get("x", tensors.get("s_t"))
        y = tensors.get("y", tensors.get("s_tp1"))
        if x is None or y is None:
            raise ValueError("Tensor dict requires `x/y` or `s_t/s_tp1`.")

        splits = tensors.get("splits")
        pair_ids = tensors.get("pair_ids")
        samples: list[dict[str, Any]] = []
        for i in range(x.shape[0]):
            sample = {
                "x": x[i],
                "y": y[i],
                "s_t": tensors.get("s_t", x)[i],
                "s_tp1": tensors.get("s_tp1", y)[i],
                "split": splits[i] if splits is not None else "mixed",
                "pair_id": pair_ids[i] if pair_ids is not None else None,
            }
            for key in [
                "r_t",
                "u_t",
                "h_slow",
                "h_fast",
                "payload_mass",
                "payload_com_shift",
                "payload_inertia",
                "load_regime",
            ]:
                if key in tensors:
                    sample[key] = tensors[key][i]
            samples.append(sample)
        return cls(samples=samples, mode=mode)

    def pair_groups(self) -> dict[str, list[int]]:
        groups: dict[str, list[int]] = {}
        for local_idx, ds_idx in enumerate(self.indices):
            pair_id = self.samples[ds_idx].pair_id
            if pair_id is None:
                continue
            groups.setdefault(pair_id, []).append(local_idx)
        return groups


def collate_world_model_samples(batch: Sequence[WorldModelSample]) -> dict[str, Any]:
    """Simple collate helper preserving structured optional fields."""

    def _stack_or_none(items: list[Any]) -> Any:
        if not items or any(x is None for x in items):
            return None
        if isinstance(items[0], torch.Tensor):
            return torch.stack(items, dim=0)
        return items

    data: dict[str, Any] = {
        "x": torch.stack([b.x for b in batch], dim=0),
        "y": torch.stack([b.y for b in batch], dim=0),
        "split": [b.split for b in batch],
        "pair_id": [b.pair_id for b in batch],
    }

    for key in [
        "s_t",
        "s_tp1",
        "r_t",
        "u_t",
        "h_slow",
        "h_fast",
        "payload_mass",
        "payload_com_shift",
        "payload_inertia",
        "load_regime",
    ]:
        data[key] = _stack_or_none([getattr(b, key) for b in batch])

    return data
