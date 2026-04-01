from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class MultiScaleHistoryEncoderConfig:
    """Config for lightweight multi-scale history encoding."""

    slow_hidden_dim: int = 192
    slow_num_layers: int = 2
    slow_out_dim: int = 128
    fast_hidden_dim: int = 128
    fast_num_layers: int = 1
    fast_out_dim: int = 64
    fused_dim: int = 128
    dropout: float = 0.0
    use_layer_norm: bool = True


class _HistoryBranch(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        out_dim: int,
        dropout: float,
        use_layer_norm: bool,
    ) -> None:
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        heads: list[nn.Module] = [nn.Linear(hidden_dim, out_dim)]
        if use_layer_norm:
            heads.append(nn.LayerNorm(out_dim))
        heads.append(nn.Tanh())
        self.head = nn.Sequential(*heads)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.gru(x)
        return self.head(y[:, -1, :])


class MultiScaleHistoryEncoder(nn.Module):
    """Encode long/short proprio history into one fused latent.

    This is the simplified default path and keeps API-compatible hooks for
    future DualGRUInteractionEncoder-style upgrades.
    """

    def __init__(
        self,
        input_dim: int,
        config: MultiScaleHistoryEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or MultiScaleHistoryEncoderConfig()
        self.cfg = cfg

        self.slow_branch = _HistoryBranch(
            input_dim=input_dim,
            hidden_dim=cfg.slow_hidden_dim,
            num_layers=cfg.slow_num_layers,
            out_dim=cfg.slow_out_dim,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )
        self.fast_branch = _HistoryBranch(
            input_dim=input_dim,
            hidden_dim=cfg.fast_hidden_dim,
            num_layers=cfg.fast_num_layers,
            out_dim=cfg.fast_out_dim,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )

        fuse_layers: list[nn.Module] = [
            nn.Linear(cfg.slow_out_dim + cfg.fast_out_dim, cfg.fused_dim),
        ]
        if cfg.use_layer_norm:
            fuse_layers.append(nn.LayerNorm(cfg.fused_dim))
        fuse_layers.append(nn.Tanh())
        self.fuse = nn.Sequential(*fuse_layers)

        self.slow_out_dim = cfg.slow_out_dim
        self.fast_out_dim = cfg.fast_out_dim
        self.fused_dim = cfg.fused_dim

    @staticmethod
    def _as_seq(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected history [B,T,D] or [B,D], got {tuple(x.shape)}")
        return x

    def forward(
        self,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor,
        return_branches: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hs = self._as_seq(h_slow)
        hf = self._as_seq(h_fast)

        z_slow = self.slow_branch(hs)
        z_fast = self.fast_branch(hf)
        z_hist = self.fuse(torch.cat([z_slow, z_fast], dim=-1))

        if return_branches:
            return z_hist, z_slow, z_fast
        return z_hist
