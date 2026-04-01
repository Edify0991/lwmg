from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class DualGRUEncoderConfig:
    """Configuration for slow/fast interaction encoder branches."""

    slow_hidden_dim: int = 256
    slow_num_layers: int = 2
    slow_latent_dim: int = 128
    fast_hidden_dim: int = 128
    fast_num_layers: int = 1
    fast_latent_dim: int = 64
    dropout: float = 0.0
    use_layer_norm: bool = True


class _GRUBranch(nn.Module):
    """Single GRU branch with projection head."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int,
        output_dim: int,
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
        layers: list[nn.Module] = [nn.Linear(hidden_dim, output_dim)]
        if use_layer_norm:
            layers.append(nn.LayerNorm(output_dim))
        layers.append(nn.Tanh())
        self.proj = nn.Sequential(*layers)

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(seq)
        last = out[:, -1, :]
        return self.proj(last)


class DualGRUInteractionEncoder(nn.Module):
    """Encode persistent and transient interaction dynamics with dual GRUs.

    `z_slow` models persistent endpoint load bias (payload/posture shift), while
    `z_fast` captures rapid events (contact transitions, slip, abrupt errors).
    """

    def __init__(
        self,
        slow_input_dim: int,
        fast_input_dim: int | None = None,
        config: DualGRUEncoderConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or DualGRUEncoderConfig()
        fast_dim = fast_input_dim if fast_input_dim is not None else slow_input_dim

        self.slow_branch = _GRUBranch(
            input_dim=slow_input_dim,
            hidden_dim=cfg.slow_hidden_dim,
            num_layers=cfg.slow_num_layers,
            output_dim=cfg.slow_latent_dim,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )
        self.fast_branch = _GRUBranch(
            input_dim=fast_dim,
            hidden_dim=cfg.fast_hidden_dim,
            num_layers=cfg.fast_num_layers,
            output_dim=cfg.fast_latent_dim,
            dropout=cfg.dropout,
            use_layer_norm=cfg.use_layer_norm,
        )

        self.slow_latent_dim = cfg.slow_latent_dim
        self.fast_latent_dim = cfg.fast_latent_dim

    @staticmethod
    def _as_seq(x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            return x.unsqueeze(1)
        if x.ndim != 3:
            raise ValueError(f"Expected history tensor with shape [B, T, D] or [B, D], got {tuple(x.shape)}")
        return x

    def forward(self, h_slow: torch.Tensor, h_fast: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h_slow_seq = self._as_seq(h_slow)
        h_fast_seq = self._as_seq(h_fast)
        z_slow = self.slow_branch(h_slow_seq)
        z_fast = self.fast_branch(h_fast_seq)
        return z_slow, z_fast
