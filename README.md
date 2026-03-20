# LWMG: Load-Aware World-Model-Guided Humanoid Motion Generation

LWMG is an Isaac Lab external project for **load-aware humanoid reference generation** using **NVIDIA GR00T-WholeBodyControl / SONIC** as a **frozen whole-body tracker** for Unitree G1.

## Why SONIC is frozen in this repository

This repository intentionally **does not vendor or reimplement** the official SONIC deployment stack.
Instead, it provides:

- **Path A (Research Loop)**: a Python-side Isaac Lab adapter that loads SONIC ONNX checkpoints and observation config for frozen tracking during rollout.
- **Path B (Deployment Compatibility)**: export and optional streaming adapters that produce SONIC-compatible references for the official runtime.

Generated references are always evaluated *through a tracker* (SONIC/mock/PD) because direct playback bypasses tracking feasibility constraints that matter for deployment.

## Project layout

- `source/lwmg/lwmg`: core Python package.
- `configs/`: Hydra/YAML config files for env/tracker/world-model/guidance/train.
- `scripts/`: data collection, training, evaluation, export, and mock sim2sim demos.
- `tests/`: unit tests for adapters, losses, data, and ranking.

## Installation (Isaac Lab external project)

1. Install Isaac Lab and Isaac Sim in your Python 3.10 environment.
2. Install this project in editable mode:

```bash
python -m pip install -e source/lwmg
python -m pip install -r requirements.txt
```

## SONIC assets (download separately)

Place public SONIC files on your machine (not committed to this repo), e.g.:

```text
/path/to/sonic/
  model_encoder.onnx
  model_decoder.onnx
  planner_sonic.onnx
  observation_config.yaml
```

Configure the path in `configs/sonic/sonic_adapter.yaml`.

## Execution paths

### Path A — Isaac Lab research loop

- Unitree G1 load-randomized simulation
- Frozen SONIC tracker adapter (or PD/mock tracker)
- rollout collection
- load-aware world model training
- candidate guidance and refinement evaluation

Example commands:

```bash
python scripts/collect_rollouts.py --config configs/train/collect_rollouts.yaml
python scripts/train_world_model.py --config configs/train/train_wm.yaml
python scripts/eval_candidate_guidance.py --config configs/train/eval_guidance.yaml
```

### Path B — export/deployment compatibility

- export generated references to SONIC motion format (`50 Hz`, IsaacLab G1 joint order)
- optional ZMQ qpos stream for compatibility with official SONIC deployment pipeline

Example commands:

```bash
python scripts/export_sonic_references.py --config configs/train/export_sonic_refs.yaml
python scripts/run_mock_sim2sim.py --config configs/sonic/sonic_export.yaml
```

## Development notes

- Default frozen backend: ONNX Runtime.
- Optional TensorRT hook points exist in the runner API but are not required.
- Logging: TensorBoard + CSV.
- Type hints and dataclasses are used across interfaces.
