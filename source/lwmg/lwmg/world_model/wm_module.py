from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from .deformation_field_decoder import DeformationFieldConfig, DeformationFieldDecoder
from .load_residual_model import LoadResidualModel, LoadResidualModelConfig
from .multi_scale_history_encoder import MultiScaleHistoryEncoder, MultiScaleHistoryEncoderConfig
from .nominal_transition_model import NominalTransitionModel
from .task_costs import (
    compose_total_cost,
    deformation_cost,
    smoothness_cost,
    stability_cost,
    task_cost_balance,
    task_cost_locomotion,
    task_cost_standup,
    tracking_cost,
)


@dataclass(frozen=True)
class HistoryDims:
    slow: int
    fast: int
    fused: int


class StructuredClosedLoopWorldModel(nn.Module):
    """Simplified-but-extensible closed-loop, reference-conditioned world model.

    Default v1 path:
    - nominal transition: f_nom(s_t, r_t, u_t)
    - multi-scale history encoding: z_hist = E(h_slow, h_fast)
    - load residual prediction: delta_s_load = f_res(s_t, r_t, u_t, z_hist)
    - final prediction: s_hat = s_nom + delta_s_load

    Future hooks are kept for mode-gating/residual-experts upgrades.
    """

    def __init__(
        self,
        state_dim: int = 32,
        ref_dim: int = 29,
        ctrl_dim: int = 29,
        latent_dim: int = 128,
        *,
        slow_latent_dim: int = 128,
        fast_latent_dim: int = 64,
        history_steps_slow: int = 8,
        history_steps_fast: int = 4,
        slow_hidden_dim: int = 192,
        slow_num_layers: int = 2,
        fast_hidden_dim: int = 128,
        fast_num_layers: int = 1,
        history_dropout: float = 0.0,
        history_use_layer_norm: bool = True,
        residual_hidden_dim: int = 192,
        residual_use_control: bool = True,
        uncertainty_mode: str = "none",  # none | ensemble | mc_dropout
        uncertainty_mc_samples: int = 4,
        residual_dropout: float = 0.0,
        deformation_latent_dim: int = 16,
        deformation_num_time_basis: int = 4,
        deformation_basis_type: str = "piecewise_linear",
        deformation_max_group_scale: float = 0.20,
        deformation_channel_groups: dict[str, list[int]] | None = None,
        deformation_group_scale_limits: dict[str, float] | None = None,
        ensemble: list[nn.Module] | None = None,
        enable_aux_load_head: bool = False,
        aux_load_dim: int = 4,
        # Future stubs (kept for compatibility/upgrade path)
        enable_mode_gating_stub: bool = False,
        enable_residual_experts_stub: bool = False,
    ) -> None:
        super().__init__()
        self.state_dim = int(state_dim)
        self.ref_dim = int(ref_dim)
        self.ctrl_dim = int(ctrl_dim)
        self.fast_history_steps = int(max(1, history_steps_fast))
        self.slow_history_steps = int(max(1, history_steps_slow))

        self.nominal = NominalTransitionModel(state_dim, ref_dim, ctrl_dim)

        hist_cfg = MultiScaleHistoryEncoderConfig(
            slow_hidden_dim=slow_hidden_dim,
            slow_num_layers=slow_num_layers,
            slow_out_dim=slow_latent_dim,
            fast_hidden_dim=fast_hidden_dim,
            fast_num_layers=fast_num_layers,
            fast_out_dim=fast_latent_dim,
            fused_dim=latent_dim,
            dropout=history_dropout,
            use_layer_norm=history_use_layer_norm,
        )
        self.history_encoder = MultiScaleHistoryEncoder(input_dim=state_dim, config=hist_cfg)

        res_cfg = LoadResidualModelConfig(
            hidden_dim=residual_hidden_dim,
            num_layers=3,
            dropout=float(residual_dropout),
            use_control=bool(residual_use_control),
        )
        self.load_residual = LoadResidualModel(
            state_dim=state_dim,
            ref_dim=ref_dim,
            ctrl_dim=ctrl_dim,
            history_latent_dim=latent_dim,
            config=res_cfg,
        )

        self.deformation_decoder = DeformationFieldDecoder(
            reference_dim=ref_dim,
            config=DeformationFieldConfig(
                latent_dim=deformation_latent_dim,
                num_time_basis=deformation_num_time_basis,
                basis_type=deformation_basis_type,
                max_group_scale=deformation_max_group_scale,
            ),
            channel_groups=deformation_channel_groups,
            group_scale_limits=deformation_group_scale_limits,
        )

        self.history_dims = HistoryDims(slow=slow_latent_dim, fast=fast_latent_dim, fused=latent_dim)

        # Legacy aliases to preserve compatibility with existing scripts/tests.
        self.interaction = self.history_encoder
        self.interaction_encoder = self.history_encoder
        self.residual = self.load_residual
        self.residual_experts = self.load_residual

        # Future upgrade stubs.
        self.mode_gating = nn.Identity() if enable_mode_gating_stub else None
        self.residual_experts_stub = nn.Identity() if enable_residual_experts_stub else None

        self.uncertainty_mode = str(uncertainty_mode).strip().lower()
        self.uncertainty_mc_samples = int(max(1, uncertainty_mc_samples))
        self.ensemble = list(ensemble or [])

        self.aux_load_head = nn.Linear(latent_dim, aux_load_dim) if enable_aux_load_head else None

    @staticmethod
    def _as_history_tensor(history: torch.Tensor, feature_dim: int) -> torch.Tensor:
        if history.ndim == 2:
            history = history.unsqueeze(1)
        if history.ndim != 3:
            raise ValueError(f"history must be [B,T,D] or [B,D], got {tuple(history.shape)}")
        if history.shape[-1] != feature_dim:
            raise ValueError(f"history feature mismatch: expected={feature_dim}, got={history.shape[-1]}")
        return history

    def _prepare_histories(self, h_slow: torch.Tensor | None, h_fast: torch.Tensor | None, s_ref: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if h_slow is None:
            h_slow_t = s_ref.unsqueeze(1)
        else:
            h_slow_t = self._as_history_tensor(h_slow, self.state_dim)

        if h_fast is None:
            h_fast_t = h_slow_t[:, -min(self.fast_history_steps, h_slow_t.shape[1]) :, :]
        else:
            h_fast_t = self._as_history_tensor(h_fast, self.state_dim)
        return h_slow_t, h_fast_t

    @staticmethod
    def _append_history(history: torch.Tensor, value: torch.Tensor, max_steps: int) -> torch.Tensor:
        return torch.cat([history, value.unsqueeze(1)], dim=1)[:, -max_steps:, :]

    def encode_history(
        self,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        hs, hf = self._prepare_histories(h_slow, h_fast, h_slow if h_slow.ndim == 2 else h_slow[:, -1])
        z_hist, z_slow, z_fast = self.history_encoder(hs, hf, return_branches=True)
        if not return_details:
            return z_hist
        return {
            "z_hist": z_hist,
            "z_slow": z_slow,
            "z_fast": z_fast,
        }

    # Legacy compatibility alias
    def encode_interaction(
        self,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
        *_unused: torch.Tensor,
        return_tuple: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        details = self.encode_history(h_slow, h_fast, return_details=True)
        if return_tuple:
            return details["z_slow"], details["z_fast"]
        return details["z_hist"]

    def predict_nominal_step(self, s_t: torch.Tensor, r_t: torch.Tensor, u_t: torch.Tensor | None = None) -> torch.Tensor:
        if u_t is None:
            u_t = torch.zeros(s_t.shape[0], self.ctrl_dim, device=s_t.device, dtype=s_t.dtype)
        return self.nominal(s_t, r_t, u_t)

    def predict_load_delta(
        self,
        s_t: torch.Tensor,
        r_t: torch.Tensor,
        u_t: torch.Tensor | None,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        hist = self.encode_history(h_slow, h_fast, return_details=True)
        delta, debug = self.load_residual(s_t, r_t, u_t, hist["z_hist"], return_debug=True)
        if not return_details:
            return delta
        return delta, {**hist, **debug}

    # Legacy compatibility alias
    def predict_interaction_delta(
        self,
        s_t: torch.Tensor,
        r_t: torch.Tensor,
        u_t: torch.Tensor,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return self.predict_load_delta(s_t, r_t, u_t, h_slow, h_fast, return_details=return_details)

    def predict_step(
        self,
        s_t: torch.Tensor,
        r_t: torch.Tensor,
        u_t: torch.Tensor | None,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
        *,
        return_details: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
        s_nom = self.predict_nominal_step(s_t, r_t, u_t)
        delta, details = self.predict_load_delta(
            s_t,
            r_t,
            u_t,
            h_slow,
            h_fast,
            return_details=True,
        )
        s_hat = s_nom + delta
        if not return_details:
            return s_hat
        return s_hat, {
            "s_nom": s_nom,
            "delta_s_load": delta,
            "delta_s_int": delta,  # compatibility key
            **details,
        }

    def step(
        self,
        s_t: torch.Tensor,
        r_t: torch.Tensor,
        u_t: torch.Tensor | None,
        h_slow: torch.Tensor,
        h_fast: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.predict_step(s_t, r_t, u_t, h_slow, h_fast)

    def rollout(
        self,
        s0: torch.Tensor,
        references: torch.Tensor,
        controls: torch.Tensor | None,
        h_slow: torch.Tensor | None,
        h_fast: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hs, hf = self._prepare_histories(h_slow, h_fast, s0)
        if controls is None:
            controls = torch.zeros(references.shape[0], references.shape[1], self.ctrl_dim, device=references.device, dtype=references.dtype)

        slow_steps, fast_steps = hs.shape[1], hf.shape[1]
        states = [s0]
        s_t = s0

        for t in range(references.shape[1]):
            s_t = self.predict_step(s_t, references[:, t], controls[:, t], hs, hf)
            states.append(s_t)
            hs = self._append_history(hs, s_t, slow_steps)
            hf = self._append_history(hf, s_t, fast_steps)

        return torch.stack(states, dim=1)

    def predict_uncertainty(
        self,
        s0: torch.Tensor,
        references: torch.Tensor,
        controls: torch.Tensor | None,
        h_slow: torch.Tensor | None,
        h_fast: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.ensemble:
            preds = torch.stack([m.rollout(s0, references, controls, h_slow, h_fast) for m in self.ensemble], dim=0)
            return preds.var(dim=0, unbiased=False)

        if self.uncertainty_mode != "mc_dropout":
            pred = self.rollout(s0, references, controls, h_slow, h_fast)
            return torch.zeros_like(pred)

        was_training = self.training
        self.train(True)
        preds = []
        for _ in range(self.uncertainty_mc_samples):
            preds.append(self.rollout(s0, references, controls, h_slow, h_fast))
        preds_t = torch.stack(preds, dim=0)
        self.train(was_training)
        return preds_t.var(dim=0, unbiased=False)

    def predict_aux_load(self, z_hist: torch.Tensor) -> torch.Tensor | None:
        if self.aux_load_head is None:
            return None
        return self.aux_load_head(z_hist)

    def deform_reference(
        self,
        nominal_reference: torch.Tensor,
        z_def: torch.Tensor,
        active_groups: list[str] | None = None,
    ) -> torch.Tensor:
        return self.deformation_decoder(nominal_reference, z_def, active_groups=active_groups)

    def score_reference(
        self,
        *,
        reference: torch.Tensor,
        rollout_states: torch.Tensor,
        nominal_reference: torch.Tensor | None = None,
        delta_reference: torch.Tensor | None = None,
        task: str = "balance",
        goal: dict[str, float] | None = None,
        weights: dict[str, float] | None = None,
        support_margin: torch.Tensor | None = None,
        stance_slip: torch.Tensor | None = None,
        uncertainty: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        goal = goal or {}
        if task == "standup":
            task_term = task_cost_standup(
                rollout_states,
                target_root_height=float(goal.get("target_root_height", 0.82)),
                terminal_upright_target=float(goal.get("terminal_upright_target", 0.0)),
            )
        elif task == "locomotion":
            task_term = task_cost_locomotion(
                rollout_states,
                target_planar_vel=tuple(goal.get("target_planar_vel", (0.5, 0.0))),
            )
        else:
            task_term = task_cost_balance(
                rollout_states,
                target_root_height=float(goal.get("target_root_height", 0.78)),
            )

        nominal = nominal_reference if nominal_reference is not None else reference
        delta = delta_reference if delta_reference is not None else (reference - nominal)

        trk = tracking_cost(nominal, reference)
        stab = stability_cost(
            support_margin,
            stance_slip,
            device=rollout_states.device,
            dtype=rollout_states.dtype,
        )
        smt = smoothness_cost(reference)
        deform = deformation_cost(delta)

        total, breakdown = compose_total_cost(
            task=task_term,
            tracking=trk,
            stability=stab,
            smoothness=smt,
            deformation=deform,
            uncertainty=uncertainty,
            weights=weights,
        )
        breakdown["total"] = total
        return breakdown

    def rollout_from_reference(self, reference: torch.Tensor) -> dict[str, torch.Tensor]:
        # Keep compatibility with existing flow-guidance utility.
        trunk_tilt = torch.abs(reference[..., 0])
        base_height = 0.8 - 0.1 * torch.abs(reference[..., 1])
        tracking_error = torch.abs(reference).mean(dim=-1)
        support_margin = 0.2 - 0.05 * torch.abs(reference[..., 2])
        slip = torch.abs(reference[:, 1:, :2] - reference[:, :-1, :2]).mean(dim=-1)
        torque = torch.abs(reference[..., :6])
        return {
            "task_progress": torch.sigmoid(reference[..., 0].mean(dim=1)),
            "target_vel_error": torch.abs(reference[..., 1].mean(dim=1)),
            "tracking_error": tracking_error,
            "trunk_tilt": trunk_tilt,
            "base_height": base_height,
            "support_margin": support_margin,
            "slip": slip,
            "torque": torque,
            "uncertainty": reference.var(dim=-1, unbiased=False),
        }
