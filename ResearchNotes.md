# Research Notes: LWMG

## Stage 0: create Isaac Lab external project
- Project skeleton, configs, scripts, tests.

## Stage 1: tracker baselines
- Mock tracker and PD fallback.

## Stage 2: SONIC frozen tracker integration
- ONNX encoder/decoder + SONIC observation config parsing.

## Stage 3: randomized load rollout collection
- reference -> SONIC -> physics pipeline under payload randomization.

## Stage 4: structured closed-loop world model
- Nominal transition + interaction encoder + residual transition.
- Reference-conditioned closed-loop prediction under loading.

## Stage 4.1: simplified publishable v1
- Keep flow prior + frozen SONIC + closed-loop rollout.
- Upgrade WM to nominal + load residual decomposition:
  - `s_nom_{t+1} = f_nom(s_t, r_t, u_t)`
  - `z_hist = E(h_slow, h_fast)`
  - `delta_s_load = f_res(s_t, r_t, u_t, z_hist)`
  - `s_hat_{t+1} = s_nom_{t+1} + delta_s_load`
- Add structured low-dimensional deformation decoder:
  - `r_star = r_nom + D(z_def)`
  - channel-group masking + time basis expansion
- Add test-time latent optimization over `z_def` using WM rollout costs.
- Add paired counterfactual support (`same nominal ref + different load`) in data path.
- Use support-consistency in v1; richer energy/interaction losses optional.

## Stage 5: flow ODE internal guidance
- Flow Matching remains nominal prior.
- ODE internal guidance preserved as compatible optional path.
- Default stable adaptation path is latent deformation optimization.

## Stage 6: future fast flow families
- Rectified Flow / Mean Flow drop-in replacements for faster real-time deployment.

## Stage 7: SONIC export / deployment compatibility
- SONIC motion clip export + optional ZMQ streaming.

## Future extensions (already hooked in code/config)
- Dual interaction encoders as primary path.
- Contact-mode gating + residual experts.
- Richer physical regularization (energy consistency).
- Internal flow-time guidance reusing WM scoring utilities.
