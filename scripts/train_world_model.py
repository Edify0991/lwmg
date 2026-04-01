from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from lwmg.data.dataset_wm import WMDataset, collate_world_model_samples
from lwmg.world_model.losses import (
    auxiliary_load_regression_loss,
    deformation_penalty_loss,
    energy_consistency_loss,
    nominal_rollout_loss,
    nominal_state_loss,
    paired_counterfactual_loss,
    residual_rollout_loss,
    residual_state_loss,
    residual_zero_loss,
    support_consistency_loss,
    uncertainty_regularization_loss,
)
from lwmg.world_model.wm_module import StructuredClosedLoopWorldModel


def _dict_get(d: dict[str, Any], key: str, default: Any) -> Any:
    val = d.get(key, default)
    return default if val is None else val


def _loss_switch(loss_cfg: dict[str, Any], key: str, default_weight: float = 0.0) -> tuple[bool, float]:
    item = loss_cfg.get(key, {})
    if isinstance(item, dict):
        return bool(item.get("enabled", False)), float(item.get("weight", default_weight))
    if isinstance(item, (int, float)):
        return True, float(item)
    return False, 0.0


def _build_model(cfg: dict[str, Any]) -> StructuredClosedLoopWorldModel:
    wm_cfg = cfg.get("world_model", {})
    hist_cfg = wm_cfg.get("multi_scale_history_encoder", {})
    residual_cfg = wm_cfg.get("load_residual_model", {})
    deform_cfg = wm_cfg.get("deformation_field_decoder", {})
    uncertainty_cfg = wm_cfg.get("uncertainty", {})
    mode_stub_cfg = wm_cfg.get("mode_gating_stub", {})
    experts_stub_cfg = wm_cfg.get("residual_experts_stub", {})
    aux_cfg = wm_cfg.get("auxiliary_load_head", {})

    return StructuredClosedLoopWorldModel(
        state_dim=int(_dict_get(wm_cfg, "state_dim", 32)),
        ref_dim=int(_dict_get(wm_cfg, "ref_dim", 29)),
        ctrl_dim=int(_dict_get(wm_cfg, "ctrl_dim", 29)),
        latent_dim=int(_dict_get(hist_cfg, "fused_dim", 128)),
        slow_latent_dim=int(_dict_get(hist_cfg, "slow_out_dim", 128)),
        fast_latent_dim=int(_dict_get(hist_cfg, "fast_out_dim", 64)),
        history_steps_slow=int(_dict_get(hist_cfg, "h_slow_len", 8)),
        history_steps_fast=int(_dict_get(hist_cfg, "h_fast_len", 4)),
        slow_hidden_dim=int(_dict_get(hist_cfg, "slow_hidden_dim", 192)),
        slow_num_layers=int(_dict_get(hist_cfg, "slow_num_layers", 2)),
        fast_hidden_dim=int(_dict_get(hist_cfg, "fast_hidden_dim", 128)),
        fast_num_layers=int(_dict_get(hist_cfg, "fast_num_layers", 1)),
        history_dropout=float(_dict_get(hist_cfg, "dropout", 0.0)),
        history_use_layer_norm=bool(_dict_get(hist_cfg, "use_layer_norm", True)),
        residual_hidden_dim=int(_dict_get(residual_cfg, "hidden_dim", 192)),
        residual_use_control=bool(_dict_get(residual_cfg, "use_control", True)),
        residual_dropout=float(_dict_get(residual_cfg, "dropout", 0.0)),
        uncertainty_mode=str(_dict_get(uncertainty_cfg, "mode", "none")),
        uncertainty_mc_samples=int(_dict_get(uncertainty_cfg, "mc_samples", 4)),
        deformation_latent_dim=int(_dict_get(deform_cfg, "latent_dim", 16)),
        deformation_num_time_basis=int(_dict_get(deform_cfg, "num_time_basis", 4)),
        deformation_basis_type=str(_dict_get(deform_cfg, "basis_type", "piecewise_linear")),
        deformation_max_group_scale=float(_dict_get(deform_cfg, "max_group_scale", 0.20)),
        enable_aux_load_head=bool(_dict_get(aux_cfg, "enabled", False)),
        aux_load_dim=int(_dict_get(aux_cfg, "out_dim", 4)),
        enable_mode_gating_stub=bool(_dict_get(mode_stub_cfg, "enabled", False)),
        enable_residual_experts_stub=bool(_dict_get(experts_stub_cfg, "enabled", False)),
    )


def _set_train_stage(model: StructuredClosedLoopWorldModel, stage: str) -> None:
    for p in model.parameters():
        p.requires_grad = False

    if stage == "nominal":
        for p in model.nominal.parameters():
            p.requires_grad = True
        return

    if stage == "residual":
        for module in [model.history_encoder, model.load_residual]:
            for p in module.parameters():
                p.requires_grad = True
        if model.aux_load_head is not None:
            for p in model.aux_load_head.parameters():
                p.requires_grad = True
        return

    # joint
    for p in model.parameters():
        p.requires_grad = True


def _build_synthetic_loader(
    batch_size: int,
    n_batches: int,
    state_dim: int,
    ref_dim: int,
    ctrl_dim: int,
    h_slow_len: int,
    h_fast_len: int,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for _ in range(n_batches):
        s_t = torch.randn(batch_size, state_dim)
        r_t = torch.randn(batch_size, ref_dim)
        u_t = torch.randn(batch_size, ctrl_dim)

        h_slow = torch.randn(batch_size, h_slow_len, state_dim)
        h_fast = h_slow[:, -h_fast_len:, :]

        s_nom = s_t + 0.05 * torch.tanh(torch.randn_like(s_t))
        load_delta = 0.03 * torch.randn_like(s_t)

        split = ["nominal"] * (batch_size // 2) + ["loaded"] * (batch_size - batch_size // 2)
        s_tp1 = s_nom.clone()
        s_tp1[batch_size // 2 :] = s_tp1[batch_size // 2 :] + load_delta[batch_size // 2 :]

        pair_id = [None] * batch_size
        for i in range(batch_size // 2):
            j = i + batch_size // 2
            if j < batch_size:
                pair_id[i] = f"p{i}"
                pair_id[j] = f"p{i}"

        out.append(
            {
                "x": s_t,
                "y": s_tp1,
                "s_t": s_t,
                "s_tp1": s_tp1,
                "r_t": r_t,
                "u_t": u_t,
                "h_slow": h_slow,
                "h_fast": h_fast,
                "split": split,
                "pair_id": pair_id,
                "support_margin": torch.rand(batch_size) * 0.04,
                "stance_foot_slip": torch.randn(batch_size) * 0.04,
                "load_regime": [0 if s == "nominal" else 1 for s in split],
                "payload_mass": torch.rand(batch_size, 1),
                "delta_reference": torch.zeros(batch_size, 1, ref_dim),
            }
        )
    return out


def _resolve_batch(batch: dict[str, Any], model: StructuredClosedLoopWorldModel) -> dict[str, Any]:
    s_t = batch.get("s_t", batch["x"])
    s_tp1 = batch.get("s_tp1", batch["y"])
    bsz = s_t.shape[0]

    r_t = batch.get("r_t")
    if r_t is None:
        r_t = torch.zeros(bsz, model.ref_dim, device=s_t.device, dtype=s_t.dtype)

    u_t = batch.get("u_t")
    if u_t is None:
        u_t = torch.zeros(bsz, model.ctrl_dim, device=s_t.device, dtype=s_t.dtype)

    h_slow = batch.get("h_slow", s_t.unsqueeze(1))
    h_fast = batch.get("h_fast", h_slow[:, -min(model.fast_history_steps, h_slow.shape[1]) :, :])

    return {
        "s_t": s_t,
        "s_tp1": s_tp1,
        "r_t": r_t,
        "u_t": u_t,
        "h_slow": h_slow,
        "h_fast": h_fast,
        "split": batch.get("split", ["mixed"] * bsz),
        "pair_id": batch.get("pair_id", [None] * bsz),
        "support_margin": batch.get("support_margin"),
        "stance_foot_slip": batch.get("stance_foot_slip"),
        "external_wrench": batch.get("external_wrench"),
        "load_regime": batch.get("load_regime"),
        "payload_mass": batch.get("payload_mass"),
        "delta_reference": batch.get("delta_reference"),
    }


def _pair_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    pair_id: list[str | None],
    split: list[str],
) -> torch.Tensor:
    groups: dict[str, list[int]] = defaultdict(list)
    for i, pid in enumerate(pair_id):
        if pid is not None:
            groups[str(pid)].append(i)

    losses = []
    for idxs in groups.values():
        if len(idxs) < 2:
            continue
        load_idx = [i for i in idxs if split[i] in {"load", "loaded"}]
        nom_idx = [i for i in idxs if split[i] == "nominal"]
        if load_idx and nom_idx:
            i0, i1 = load_idx[0], nom_idx[0]
        else:
            i0, i1 = idxs[0], idxs[1]
        losses.append(
            paired_counterfactual_loss(
                pred_loaded=pred[i0 : i0 + 1],
                pred_nominal=pred[i1 : i1 + 1],
                true_loaded=true[i0 : i0 + 1],
                true_nominal=true[i1 : i1 + 1],
            )
        )
    if not losses:
        return torch.zeros((), device=pred.device, dtype=pred.dtype)
    return torch.stack(losses).mean()


def _to_regime_tensor(load_regime: Any, device: torch.device) -> torch.Tensor | None:
    if load_regime is None:
        return None
    if isinstance(load_regime, torch.Tensor):
        return load_regime.to(device=device)
    if isinstance(load_regime, list):
        return torch.tensor([int(x) for x in load_regime], device=device, dtype=torch.long)
    return None


def _resolve_dataset_mode(stage: str, data_cfg: dict[str, Any]) -> str:
    groups = data_cfg.get("groups", {})
    if stage == "nominal":
        return "nominal"
    if not isinstance(groups, dict):
        return "mixed"

    use_nom = bool(groups.get("nominal", False))
    use_load = bool(groups.get("load", False))
    use_pair = bool(groups.get("pair", False))

    enabled = sum([use_nom, use_load, use_pair])
    if enabled != 1:
        return "mixed"
    if use_nom:
        return "nominal"
    if use_load:
        return "load"
    return "pair"


def _compute_loss(model: StructuredClosedLoopWorldModel, batch: dict[str, Any], stage: str, loss_cfg: dict[str, Any]) -> torch.Tensor:
    data = _resolve_batch(batch, model)
    s_t = data["s_t"]
    s_tp1 = data["s_tp1"]
    r_t = data["r_t"]
    u_t = data["u_t"]
    h_slow = data["h_slow"]
    h_fast = data["h_fast"]

    s_nom = model.predict_nominal_step(s_t, r_t, u_t)
    pred, details = model.predict_step(s_t, r_t, u_t, h_slow, h_fast, return_details=True)

    total = torch.zeros((), device=s_t.device, dtype=s_t.dtype)

    nom_state_on, nom_state_w = _loss_switch(loss_cfg, "nominal_state", 1.0)
    nom_roll_on, nom_roll_w = _loss_switch(loss_cfg, "nominal_rollout", 1.0)
    res_state_on, res_state_w = _loss_switch(loss_cfg, "residual_state", 1.0)
    res_roll_on, res_roll_w = _loss_switch(loss_cfg, "residual_rollout", 1.0)
    res_zero_on, res_zero_w = _loss_switch(loss_cfg, "residual_zero", 0.2)
    pair_on, pair_w = _loss_switch(loss_cfg, "paired_counterfactual", 1.0)
    support_on, support_w = _loss_switch(loss_cfg, "support_consistency", 0.1)
    deform_on, deform_w = _loss_switch(loss_cfg, "deformation_penalty", 0.1)
    aux_load_on, aux_load_w = _loss_switch(loss_cfg, "auxiliary_load_regression", 0.0)
    unc_on, unc_w = _loss_switch(loss_cfg, "uncertainty_regularization", 0.0)
    energy_on, energy_w = _loss_switch(loss_cfg, "energy_consistency", 0.0)  # optional hook only

    if stage in {"nominal", "joint"}:
        if nom_state_on:
            total = total + nom_state_w * nominal_state_loss(s_nom, s_tp1)
        if nom_roll_on:
            total = total + nom_roll_w * nominal_rollout_loss(s_nom.unsqueeze(1), s_tp1.unsqueeze(1))

    if stage in {"residual", "joint"}:
        target_delta = s_tp1 - s_nom.detach()
        if res_state_on:
            total = total + res_state_w * residual_state_loss(details["delta_s_load"], target_delta)
        if res_roll_on:
            total = total + res_roll_w * residual_rollout_loss(pred.unsqueeze(1), s_tp1.unsqueeze(1))
        if res_zero_on:
            nominal_mask = torch.tensor([s == "nominal" for s in data["split"]], device=s_t.device, dtype=torch.bool)
            if nominal_mask.any():
                total = total + res_zero_w * residual_zero_loss(details["delta_s_load"][nominal_mask])
        if pair_on:
            total = total + pair_w * _pair_loss(pred, s_tp1, data["pair_id"], data["split"])
        if support_on:
            total = total + support_w * support_consistency_loss(data["support_margin"], data["stance_foot_slip"])
        if deform_on and data["delta_reference"] is not None:
            total = total + deform_w * deformation_penalty_loss(data["delta_reference"])
        if aux_load_on and model.aux_load_head is not None and data["payload_mass"] is not None:
            aux_pred = model.predict_aux_load(details["z_hist"])
            total = total + aux_load_w * auxiliary_load_regression_loss(aux_pred, data["payload_mass"], enabled=True)
        if unc_on:
            unc = model.predict_uncertainty(s_t, r_t.unsqueeze(1), u_t.unsqueeze(1), h_slow, h_fast)
            total = total + unc_w * uncertainty_regularization_loss(unc, enabled=True)
        if energy_on:
            total = total + energy_w * energy_consistency_loss(s_t, pred, u_t, external_wrench=data["external_wrench"])

    return total


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", type=str, choices=["nominal", "residual", "joint"], default=None)
    parser.add_argument("--dataset", type=Path, default=None)
    args = parser.parse_args()

    cfg = yaml.safe_load(args.config.read_text())
    train_cfg = cfg.get("train", {})
    stage = args.stage or train_cfg.get("stage", "nominal")

    model = _build_model(cfg)
    _set_train_stage(model, stage)

    lr_map = {
        "nominal": float(train_cfg.get("nominal_lr", train_cfg.get("lr", 1.0e-3))),
        "residual": float(train_cfg.get("residual_lr", train_cfg.get("lr", 1.0e-3))),
        "joint": float(train_cfg.get("joint_lr", train_cfg.get("lr", 1.0e-4))),
    }
    opt = torch.optim.Adam([p for p in model.parameters() if p.requires_grad], lr=lr_map[stage])

    batch_size = int(train_cfg.get("batch_size", 16))
    epochs = int(train_cfg.get("epochs", 3))
    n_batches = int(train_cfg.get("synthetic_batches", 8))

    hist_cfg = cfg.get("world_model", {}).get("multi_scale_history_encoder", {})
    h_slow_len = int(hist_cfg.get("h_slow_len", 8))
    h_fast_len = int(hist_cfg.get("h_fast_len", 4))
    data_cfg = cfg.get("data", {})

    if args.dataset is not None and args.dataset.exists():
        data_obj = torch.load(args.dataset, map_location="cpu")
        dataset_mode = _resolve_dataset_mode(stage=stage, data_cfg=data_cfg)
        dataset = WMDataset.from_tensor_dict(data_obj, mode=dataset_mode)
        loader: Any = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_world_model_samples)
    else:
        loader = _build_synthetic_loader(
            batch_size=batch_size,
            n_batches=n_batches,
            state_dim=model.state_dim,
            ref_dim=model.ref_dim,
            ctrl_dim=model.ctrl_dim,
            h_slow_len=h_slow_len,
            h_fast_len=h_fast_len,
        )

    loss_cfg = cfg.get("world_model", {}).get("losses", {})

    for epoch in range(epochs):
        accum = 0.0
        n = 0
        for batch in loader:
            loss = _compute_loss(model, batch, stage=stage, loss_cfg=loss_cfg)
            opt.zero_grad()
            loss.backward()
            opt.step()
            accum += float(loss.detach().item())
            n += 1
        print(f"[train_world_model] epoch={epoch+1}/{epochs} stage={stage} loss={accum/max(1,n):.6f}")

    print(f"world model training complete, stage={stage}")


if __name__ == "__main__":
    main()
