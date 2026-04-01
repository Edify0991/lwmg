from __future__ import annotations

import argparse
import os
import re
from pathlib import Path
from typing import Any

import torch
import yaml

from lwmg.guidance.ode_guidance import FlowODEGuidance
from lwmg.references.flows.flow_matching_generator import FlowMatchingGenerator
from lwmg.world_model import LatentReferenceOptimizer, LatentReferenceOptimizerConfig
from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel

_OC_ENV_PATTERN = re.compile(r"^\$\{oc\.env:([^,}]+),\s*(.+)\}$")


def _metrics(ref: torch.Tensor, wm: StructuredClosedLoopWorldModel) -> dict[str, float]:
    roll = wm.rollout_from_reference(ref)
    return {
        "task_progress": float(roll["task_progress"].mean()),
        "tracking_error": float(roll["tracking_error"].mean()),
        "stability_tilt": float(roll["trunk_tilt"].mean()),
        "torque_proxy": float(roll["torque"].mean()),
        "uncertainty": float(roll["uncertainty"].mean()),
    }


def _cfg_get(cfg: dict[str, Any], path: list[str], default: Any) -> Any:
    cur: Any = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _resolve_oc_env(value: Any) -> Any:
    if not isinstance(value, str):
        return value
    match = _OC_ENV_PATTERN.match(value.strip())
    if match is None:
        return value
    env_name = match.group(1).strip()
    default = match.group(2).strip()
    if default.endswith("}"):
        default = default[:-1].strip()
    return os.environ.get(env_name, default)


def _resolve_path(raw_path: Any, base_dir: Path) -> Path:
    raw = _resolve_oc_env(raw_path)
    p = Path(str(raw)).expanduser()
    if p.is_absolute():
        return p
    by_cfg = (base_dir / p).resolve()
    if by_cfg.exists():
        return by_cfg
    return (Path.cwd() / p).resolve()


def _build_wm(cfg: dict[str, Any]) -> StructuredClosedLoopWorldModel:
    wm_cfg = cfg.get("world_model", {})
    hist_cfg = wm_cfg.get("multi_scale_history_encoder", {})
    residual_cfg = wm_cfg.get("load_residual_model", {})
    deform_cfg = wm_cfg.get("deformation_field_decoder", {})
    unc_cfg = wm_cfg.get("uncertainty", {})

    return StructuredClosedLoopWorldModel(
        state_dim=int(_cfg_get(cfg, ["world_model", "state_dim"], 32)),
        ref_dim=int(_cfg_get(cfg, ["world_model", "ref_dim"], 29)),
        ctrl_dim=int(_cfg_get(cfg, ["world_model", "ctrl_dim"], 29)),
        latent_dim=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "fused_dim"], 128)),
        slow_latent_dim=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "slow_out_dim"], 128)),
        fast_latent_dim=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "fast_out_dim"], 64)),
        history_steps_slow=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "h_slow_len"], 8)),
        history_steps_fast=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "h_fast_len"], 4)),
        slow_hidden_dim=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "slow_hidden_dim"], 192)),
        slow_num_layers=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "slow_num_layers"], 2)),
        fast_hidden_dim=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "fast_hidden_dim"], 128)),
        fast_num_layers=int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "fast_num_layers"], 1)),
        history_dropout=float(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "dropout"], 0.0)),
        history_use_layer_norm=bool(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "use_layer_norm"], True)),
        residual_hidden_dim=int(_cfg_get(cfg, ["world_model", "load_residual_model", "hidden_dim"], 192)),
        residual_use_control=bool(_cfg_get(cfg, ["world_model", "load_residual_model", "use_control"], True)),
        residual_dropout=float(_cfg_get(cfg, ["world_model", "load_residual_model", "dropout"], 0.0)),
        uncertainty_mode=str(_cfg_get(cfg, ["world_model", "uncertainty", "mode"], "none")),
        uncertainty_mc_samples=int(_cfg_get(cfg, ["world_model", "uncertainty", "mc_samples"], 4)),
        deformation_latent_dim=int(_cfg_get(cfg, ["world_model", "deformation_field_decoder", "latent_dim"], 16)),
        deformation_num_time_basis=int(_cfg_get(cfg, ["world_model", "deformation_field_decoder", "num_time_basis"], 4)),
        deformation_basis_type=str(_cfg_get(cfg, ["world_model", "deformation_field_decoder", "basis_type"], "piecewise_linear")),
        deformation_max_group_scale=float(_cfg_get(cfg, ["world_model", "deformation_field_decoder", "max_group_scale"], 0.20)),
        enable_mode_gating_stub=bool(_cfg_get(cfg, ["world_model", "mode_gating_stub", "enabled"], False)),
        enable_residual_experts_stub=bool(_cfg_get(cfg, ["world_model", "residual_experts_stub", "enabled"], False)),
    )


def _run_latent_deformation_path(
    cfg: dict[str, Any],
    batch_size: int,
    horizon: int,
    tracker_cfg_path: Path | None = None,
    run_tracker_step: bool = False,
) -> None:
    flow = FlowMatchingGenerator(
        context_dim=int(_cfg_get(cfg, ["flow_prior", "context_dim"], 16)),
        ref_dim=int(_cfg_get(cfg, ["world_model", "ref_dim"], 29)),
    )
    wm = _build_wm(cfg)

    context = torch.randn(batch_size, int(_cfg_get(cfg, ["flow_prior", "context_dim"], 16)))
    nominal = flow.sample_nominal_reference(batch_size, horizon, context)

    opt_cfg = LatentReferenceOptimizerConfig(
        num_steps=int(_cfg_get(cfg, ["optimizer", "num_steps"], 16)),
        lr=float(_cfg_get(cfg, ["optimizer", "lr"], 5.0e-2)),
        weight_decay=float(_cfg_get(cfg, ["optimizer", "weight_decay"], 0.0)),
        grad_clip_norm=float(_cfg_get(cfg, ["optimizer", "grad_clip_norm"], 5.0)),
        latent_clip=float(_cfg_get(cfg, ["optimizer", "latent_clip"], 3.0)),
    )
    optimizer = LatentReferenceOptimizer(wm.deformation_decoder, config=opt_cfg)

    state_dim = wm.state_dim
    slow_len = int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "h_slow_len"], 8))
    fast_len = int(_cfg_get(cfg, ["world_model", "multi_scale_history_encoder", "h_fast_len"], 4))

    s0 = torch.zeros(batch_size, state_dim)
    h_slow = torch.zeros(batch_size, slow_len, state_dim)
    h_fast = torch.zeros(batch_size, fast_len, state_dim)
    controls = torch.zeros(batch_size, horizon, wm.ctrl_dim)

    nominal_roll = wm.rollout(s0, nominal, controls, h_slow, h_fast)
    nominal_score = wm.score_reference(
        reference=nominal,
        rollout_states=nominal_roll,
        nominal_reference=nominal,
        delta_reference=torch.zeros_like(nominal),
        task=str(_cfg_get(cfg, ["costs", "task"], "balance")),
        goal=dict(_cfg_get(cfg, ["costs", "goal"], {})),
        weights=dict(_cfg_get(cfg, ["costs", "weights"], {})),
    )

    z_star, adapted, adapted_score = optimizer.optimize(
        world_model=wm,
        nominal_reference=nominal,
        current_state=s0,
        h_slow=h_slow,
        h_fast=h_fast,
        controls=controls,
        active_groups=list(_cfg_get(cfg, ["deformation", "active_groups"], [])),
        task=str(_cfg_get(cfg, ["costs", "task"], "balance")),
        goal=dict(_cfg_get(cfg, ["costs", "goal"], {})),
        cost_weights=dict(_cfg_get(cfg, ["costs", "weights"], {})),
    )

    print(
        "[eval_flow_guidance] mode=latent_deformation",
        f"nominal_total={float(nominal_score['total'].item()):.6f}",
        f"adapted_total={float(adapted_score['total'].item()):.6f}",
        f"delta_norm={float((adapted - nominal).pow(2).mean().sqrt().item()):.6f}",
        f"z_norm={float(z_star.pow(2).mean().sqrt().item()):.6f}",
    )
    print("[eval_flow_guidance] nominal_metrics:", _metrics(nominal, wm))
    print("[eval_flow_guidance] adapted_metrics:", _metrics(adapted, wm))

    if run_tracker_step:
        from lwmg.trackers.sonic_frozen_tracker_adapter import SonicFrozenTrackerAdapter

        if tracker_cfg_path is None:
            raise ValueError("--run-tracker-step requires --tracker-config")
        tracker_cfg = yaml.safe_load(tracker_cfg_path.read_text()) or {}
        if not isinstance(tracker_cfg, dict):
            raise ValueError(f"Expected mapping tracker YAML at {tracker_cfg_path}")
        paths_cfg = tracker_cfg.get("paths", {})
        adapter_cfg = tracker_cfg.get("adapter", {})
        if not isinstance(paths_cfg, dict) or not isinstance(adapter_cfg, dict):
            raise ValueError("Tracker config missing `paths` or `adapter` sections.")

        tracker = SonicFrozenTrackerAdapter(
            encoder_path=_resolve_path(paths_cfg["encoder_onnx"], tracker_cfg_path.parent),
            decoder_path=_resolve_path(paths_cfg["decoder_onnx"], tracker_cfg_path.parent),
            observation_config_path=_resolve_path(paths_cfg["sonic_obs_config"], tracker_cfg_path.parent),
            target_dim=wm.ctrl_dim,
            obs_mapping_path=_resolve_path(adapter_cfg["obs_mapping_cfg"], tracker_cfg_path.parent),
            provider="cpu",
        )
        # Lightweight sanity pass: first adapted reference step -> tracker action.
        action = tracker.track_reference(adapted[:, 0], torch.zeros(batch_size, 24))
        print(
            "[eval_flow_guidance] tracker_step",
            f"action_shape={tuple(action.shape)}",
            f"action_abs_max={float(action.abs().max().item()):.6f}",
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=("auto", "latent_deformation", "ode_internal"),
        default="auto",
        help="Use latent deformation optimization (default stable path) or legacy ODE internal guidance.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=12)
    parser.add_argument("--tracker-config", type=Path, default=None)
    parser.add_argument(
        "--run-tracker-step",
        action="store_true",
        help="Optional: run one frozen SONIC tracker forward pass on adapted reference.",
    )
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Expected mapping YAML for --config")

    mode = args.mode
    if mode == "auto":
        mode = str(_cfg_get(cfg, ["pipeline", "default_mode"], "latent_deformation"))

    if mode == "latent_deformation":
        _run_latent_deformation_path(
            cfg=cfg,
            batch_size=args.batch_size,
            horizon=args.horizon,
            tracker_cfg_path=args.tracker_config,
            run_tracker_step=args.run_tracker_step,
        )
        return

    # Legacy ODE internal guidance path (preserved for compatibility).
    b, t = args.batch_size, args.horizon
    flow = FlowMatchingGenerator(
        context_dim=int(_cfg_get(cfg, ["flow_prior", "context_dim"], 16)),
        ref_dim=int(_cfg_get(cfg, ["world_model", "ref_dim"], 29)),
    )
    wm = _build_wm(cfg)
    guidance = FlowODEGuidance(wm)

    for load_scale in [0.0, 0.5, 1.0]:
        context = torch.randn(b, int(_cfg_get(cfg, ["flow_prior", "context_dim"], 16))) * (1.0 + load_scale)
        unguided = flow.sample_unguided(b, t, context)
        anchor = unguided.detach()

        def grad_fn(x: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
            return guidance.grad_guidance(x, flow.decode_reference, anchor)

        guided = flow.sample_guided(b, t, context, grad_fn)
        print(
            f"[eval_flow_guidance] mode=ode_internal load_scale={load_scale}",
            {"unguided": _metrics(unguided, wm), "guided": _metrics(guided, wm)},
        )


if __name__ == "__main__":
    main()
