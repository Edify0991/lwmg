from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class LoadResidualModelConfig:
    hidden_dim: int = 192
    num_layers: int = 3
    dropout: float = 0.0
    use_control: bool = True


class LoadResidualModel(nn.Module):
    """Predict load-induced residual over nominal closed-loop dynamics."""

    def __init__(
        self,
        state_dim: int,
        ref_dim: int,
        ctrl_dim: int,
        history_latent_dim: int,
        config: LoadResidualModelConfig | None = None,
    ) -> None:
        super().__init__()
        cfg = config or LoadResidualModelConfig()
        self.cfg = cfg
        self.state_dim = int(state_dim)

        in_dim = state_dim + ref_dim + history_latent_dim
        self.use_control = bool(cfg.use_control)
        if self.use_control:
            in_dim += ctrl_dim

        blocks: list[nn.Module] = []
        prev = in_dim
        for _ in range(max(1, cfg.num_layers)):
            blocks.append(nn.Linear(prev, cfg.hidden_dim))
            blocks.append(nn.SiLU())
            if cfg.dropout > 0:
                blocks.append(nn.Dropout(cfg.dropout))
            prev = cfg.hidden_dim
        blocks.append(nn.Linear(prev, state_dim))
        self.net = nn.Sequential(*blocks)

    def forward(
        self,
        s_t: torch.Tensor,
        r_t: torch.Tensor,
        u_t: torch.Tensor | None,
        z_hist: torch.Tensor,
        return_debug: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        items = [s_t, r_t, z_hist]
        if self.use_control and u_t is not None:
            items.append(u_t)
        x = torch.cat(items, dim=-1)
        delta = self.net(x)
        if not return_debug:
            return delta
        return delta, {"residual_input": x}
