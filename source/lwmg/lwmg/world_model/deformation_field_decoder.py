from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DeformationFieldConfig:
    """Config for structured low-dimensional deformation decoding."""

    latent_dim: int = 16
    num_time_basis: int = 4
    basis_type: str = "piecewise_linear"  # piecewise_linear | dct | learned_linear
    max_group_scale: float = 0.20


def _bounded_range(start: int, stop: int, dim: int) -> list[int]:
    if start >= dim:
        return []
    return list(range(start, min(stop, dim)))


def _build_time_basis(horizon: int, n_basis: int, basis_type: str, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.linspace(0.0, 1.0, horizon, device=device, dtype=dtype)
    if basis_type == "dct":
        basis = [
            torch.cos(torch.pi * (k + 1) * t)
            for k in range(n_basis)
        ]
        return torch.stack(basis, dim=1)

    # piecewise_linear as stable default
    if basis_type == "piecewise_linear" or basis_type == "learned_linear":
        knots = torch.linspace(0.0, 1.0, n_basis, device=device, dtype=dtype)
        width = 1.0 / max(1, n_basis - 1)
        cols = []
        for k in range(n_basis):
            tri = torch.relu(1.0 - torch.abs(t - knots[k]) / max(width, 1.0e-6))
            cols.append(tri)
        return torch.stack(cols, dim=1)

    raise ValueError(f"Unsupported basis_type={basis_type}")


class DeformationFieldDecoder(nn.Module):
    """Structured deformation decoder over nominal references.

    The adaptation path is `r_star = r_nom + delta_r`, where `delta_r` is decoded
    from low-dimensional latent `z_def` through fixed/linear time bases and channel
    group masks. Zero latent always decodes to zero deformation.
    """

    def __init__(
        self,
        reference_dim: int,
        config: DeformationFieldConfig | None = None,
        channel_groups: dict[str, list[int]] | None = None,
        group_scale_limits: dict[str, float] | None = None,
    ) -> None:
        super().__init__()
        cfg = config or DeformationFieldConfig()
        self.reference_dim = int(reference_dim)
        self.cfg = cfg

        default_groups = {
            "pelvis_root": _bounded_range(0, 7, self.reference_dim),
            "trunk_orientation": _bounded_range(7, 10, self.reference_dim),
            "arm_hand": _bounded_range(10, 22, self.reference_dim),
            "lower_body": _bounded_range(22, min(46, self.reference_dim), self.reference_dim),
        }
        merged = channel_groups or default_groups
        filtered = {
            name: [idx for idx in idxs if 0 <= idx < self.reference_dim] for name, idxs in merged.items()
        }
        self.channel_groups = {name: idxs for name, idxs in filtered.items() if len(idxs) > 0}
        if not self.channel_groups:
            self.channel_groups = {"all_channels": list(range(self.reference_dim))}
        self.group_names = sorted(self.channel_groups.keys())

        # Per-group linear maps with zero-bias to guarantee z=0 -> delta=0
        self.group_maps = nn.ModuleDict()
        for name in self.group_names:
            out_dim = len(self.channel_groups[name]) * int(cfg.num_time_basis)
            self.group_maps[name] = nn.Linear(int(cfg.latent_dim), out_dim, bias=False)

        limits = {name: float(cfg.max_group_scale) for name in self.group_names}
        if group_scale_limits:
            for key, value in group_scale_limits.items():
                if key in limits:
                    limits[key] = float(value)
        self.group_scale_limits = limits

        if cfg.basis_type == "learned_linear":
            self.learned_basis_weight = nn.Parameter(torch.eye(cfg.num_time_basis))
        else:
            self.learned_basis_weight = None

    def _active_groups(self, active_groups: list[str] | None) -> list[str]:
        if active_groups is None:
            return self.group_names
        return [name for name in active_groups if name in self.channel_groups]

    def _decode_group(self, z_def: torch.Tensor, basis: torch.Tensor, group_name: str) -> torch.Tensor:
        coeff = self.group_maps[group_name](z_def)
        gdim = len(self.channel_groups[group_name])
        coeff = coeff.view(z_def.shape[0], self.cfg.num_time_basis, gdim)
        # [B,T,K] = [B,N,K] x [T,N]
        deform = torch.einsum("bnk,tn->btk", coeff, basis)
        limit = self.group_scale_limits[group_name]
        return torch.tanh(deform) * limit

    def decode_deformation(
        self,
        r_nom: torch.Tensor,
        z_def: torch.Tensor,
        active_groups: list[str] | None = None,
    ) -> torch.Tensor:
        if r_nom.ndim != 3:
            raise ValueError(f"r_nom must be [B,T,D], got {tuple(r_nom.shape)}")
        if z_def.ndim != 2:
            raise ValueError(f"z_def must be [B,L], got {tuple(z_def.shape)}")
        if r_nom.shape[0] != z_def.shape[0]:
            raise ValueError("Batch size mismatch between r_nom and z_def")
        if r_nom.shape[-1] != self.reference_dim:
            raise ValueError(f"Expected reference_dim={self.reference_dim}, got {r_nom.shape[-1]}")

        basis = _build_time_basis(
            horizon=r_nom.shape[1],
            n_basis=self.cfg.num_time_basis,
            basis_type=self.cfg.basis_type,
            device=r_nom.device,
            dtype=r_nom.dtype,
        )
        if self.learned_basis_weight is not None:
            basis = basis @ self.learned_basis_weight.to(device=basis.device, dtype=basis.dtype)

        delta = torch.zeros_like(r_nom)
        for group_name in self._active_groups(active_groups):
            idx = self.channel_groups[group_name]
            if not idx:
                continue
            delta[..., idx] = self._decode_group(z_def, basis, group_name)
        return delta

    def forward(
        self,
        r_nom: torch.Tensor,
        z_def: torch.Tensor,
        active_groups: list[str] | None = None,
        return_delta: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        squeeze = False
        if r_nom.ndim == 2:
            r_nom = r_nom.unsqueeze(1)
            squeeze = True

        delta = self.decode_deformation(r_nom=r_nom, z_def=z_def, active_groups=active_groups)
        r_star = r_nom + delta

        if squeeze:
            delta = delta[:, 0]
            r_star = r_star[:, 0]

        if return_delta:
            return delta, r_star
        return r_star
