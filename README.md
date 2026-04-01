# LWMG: Load-Aware World-Model-Guided Humanoid Motion Generation

LWMG is an Isaac Lab external project for Unitree G1 that keeps:

- a flow matching nominal motion prior,
- SONIC frozen tracker integration,
- reference-conditioned closed-loop world modeling,
- closed-loop rollout collection.

## Core method (v1 simplified, extensible)

1. Flow prior proposes a nominal reference:
   `r_nom = sample_nominal_reference(...)`
2. Frozen tracker executes reference semantics:
   `u_t = pi_trk(o_t, r_t)`
3. Closed-loop world model predicts load-aware dynamics:
   `s_nom_{t+1} = f_nom(s_t, r_t, u_t)`
   `z_hist = E(h_slow, h_fast)`
   `delta_s_load = f_res(s_t, r_t, u_t, z_hist)`
   `s_hat_{t+1} = s_nom_{t+1} + delta_s_load`
4. Structured reference adaptation deforms nominal reference:
   `r_star = r_nom + D(z_def)`
5. Test-time latent optimization:
   `z_def* = argmin J_wm(r_nom + D(z_def))`

The v1 default path uses latent deformation optimization, while preserving hooks for future internal flow-time guidance.

## Why this design

- Flow prior remains responsible for nominal motion manifold quality.
- WM remains reference-conditioned and closed-loop with frozen tracker semantics.
- Load adaptation is done by low-dimensional structured deformation instead of unconstrained full trajectory optimization.
- Support consistency is included in v1; richer energy and expert/gating models are preserved as optional future hooks.
- Paired counterfactual samples (same nominal reference, different load) are used to learn causal load-induced deviation.

## Deferred (kept as stubs/hooks)

- Dual interaction encoders as primary path.
- Contact-mode gating and residual experts as primary path.
- Required energy consistency loss.
- End-to-end internal flow guidance as default path.

## Why this differs from previous ranking/refinement

- Not GPC-style candidate ranking: guidance is injected during ODE integration, not only after candidate generation.
- Not LIFT-style action-conditioned finetuning: the world model is **reference-conditioned closed-loop**, modeling tracked execution under load.
- Risk heads are optional diagnostics; primary guidance uses explicit differentiable rollout costs.

## Flow family design

- `flow_matching` is the default working implementation.
- `rectified_flow` and `mean_flow` are pluggable stubs behind the same interface.
- Swapping flow families does not require changing world model or guidance APIs.

## Execution paths

### Path A — Research loop
- Generate nominal references with flow prior.
- Adapt references via latent deformation + WM rollout costs.
- Execute adapted references through frozen SONIC tracker in Isaac Lab.

### Path B — Deployment compatibility
- Export SONIC-compatible reference clips at 50 Hz and IsaacLab G1 joint order.
- Optional ZMQ bridge for deployment stack interoperability.

## Installation

```bash
python -m pip install -e source/lwmg
python -m pip install -r requirements.txt
```

## Key scripts

```bash
python scripts/train_world_model.py --config configs/train/train_wm_nominal.yaml --stage nominal
python scripts/train_world_model.py --config configs/train/train_wm_residual.yaml --stage residual
python scripts/train_flow_generator.py --config configs/train/train_flow_generator.yaml
python scripts/eval_flow_guidance.py --config configs/guidance/latent_reference_optimization.yaml --mode latent_deformation
python scripts/replay_motion.py --env-config configs/env/g1_walk_load.yaml --tracker-config configs/tracker/frozen_sonic_tracker.yaml
```
