# Research Notes: LWMG

## Stage 0: create Isaac Lab external project
- Establish package, config, scripts, and tests.
- Define interfaces for tracker/world-model/guidance.

## Stage 1: integrate mock tracker / PD tracker
- Verify rollout loop with deterministic and low-fidelity trackers.
- Confirm observation/action plumbing.

## Stage 2: integrate SONIC frozen tracker adapter
- Load SONIC ONNX + observation config from local user path.
- Run frozen inference in Isaac Lab Python loop.

## Stage 3: collect randomized-load rollouts
- Randomize payload mass, CoM shift, and hand wrenches.
- Preserve stable, near-failure windows, and hard-failure prefixes.

## Stage 4: train load-aware world model
- Train recurrent WM with failure-aware and wrench-aware heads.
- Validate latent load context separability.

## Stage 5: evaluate candidate guidance
- Candidate ranking and optional gradient refinement.
- Compare recovery, stability, and smoothness metrics.

## Stage 6: export motions to SONIC format / ZMQ
- Export SONIC-compatible reference files.
- Optional qpos stream bridge to official deployment stack.
