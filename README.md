# LWMG: Load-Aware World-Model-Guided Humanoid Motion Generation

LWMG is an Isaac Lab external project for Unitree G1 that uses **SONIC as a frozen whole-body tracker** and upgrades guidance from candidate ranking to **flow ODE internal guidance**.

## Core method (updated)

1. A flow-family generator (default: Flow Matching) proposes latent trajectory dynamics.
2. The latent trajectory is decoded into a reference motion.
3. A **reference-conditioned closed-loop world model** predicts tracked execution outcomes (`reference -> SONIC -> physics`).
4. A differentiable feasibility objective is backpropagated into the flow ODE dynamics:

`dx/dτ = v_θ(x, τ | h, g) - λ ∇_x J_wm(x)`

5. The final guided reference is executed by frozen SONIC.

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
- Generate reference trajectories with flow ODE internal guidance.
- Roll out closed-loop feasibility via structured world model.
- Execute references through frozen SONIC tracker in Isaac Lab.

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
python scripts/eval_flow_guidance.py --config configs/train/eval_flow_guidance.yaml
python scripts/replay_motion.py --env-config configs/env/g1_walk_load.yaml --tracker-config configs/tracker/frozen_sonic_tracker.yaml
```
