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

## Stage 5: flow ODE internal guidance
- Default Flow Matching generator.
- Guidance injected inside ODE with differentiable world-model feasibility costs.

## Stage 6: future fast flow families
- Rectified Flow / Mean Flow drop-in replacements for faster real-time deployment.

## Stage 7: SONIC export / deployment compatibility
- SONIC motion clip export + optional ZMQ streaming.
