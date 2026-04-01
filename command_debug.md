# G1 + Frozen SONIC 完整调试版清单

> 目标：覆盖导出、回放、多模式、A/B对齐、训练采集全流程。

## A. 环境与检查

```bash
cd /home/user/wmd/lwmg
conda activate wmd_isaaclab
```

```bash
ls -l \
  configs/env/g1_squat_load.yaml \
  configs/tracker/sonic_dds_bridge.yaml \
  configs/tracker/frozen_sonic_tracker.yaml \
  configs/sonic/sonic_export.yaml
```

如果普通 `python scripts/...` 找不到 `isaaclab.app`：

```bash
cd /home/user/wmd/IsaacLab
./isaaclab.sh -p /home/user/wmd/lwmg/scripts/replay_motion.py --help
```

## B. 参考动作导出（多数据源）

### B1. 导出

```bash
cd /home/user/wmd/lwmg
python scripts/export_sonic_references.py --config configs/sonic/sonic_export.yaml
```

### B2. 检查导出质量

```bash
ls -lah outputs/sonic_refs
cat outputs/sonic_refs/clips_manifest.csv
```

```bash
python - <<'PY'
import numpy as np, pathlib
clip = pathlib.Path('outputs/sonic_refs/clip_000')
for n in ['joint_pos.csv', 'joint_vel.csv', 'body_pos.csv', 'body_quat.csv']:
    a = np.loadtxt(clip / n, delimiter=',')
    print(f"{n}: shape={a.shape}, abs_max={np.abs(a).max():.6f}")
PY
```

### B3. source.type 可选

1. `lafan1_retarget_csv`
2. `amass_retarget_csv`
3. `generic_csv`
4. `generated_npz`
5. `generated_pt`
6. `synthetic_zero`（仅调试）

## C. 官方 C++ 推理 + IsaacLab replay（DDS）

### C1. 终端A：启动 C++ deploy

```bash
cd /home/user/wmd/lwmg/thirdparty/GR00T-WholeBodyControl/gear_sonic_deploy
./deploy.sh \
  --cp policy/release/model \
  --obs-config policy/release/observation_config.yaml \
  --motion-data /home/user/wmd/lwmg/outputs/sonic_refs \
  --input-type zmq_manager \
  --output-type all \
  --zmq-host 127.0.0.1 \
  sim
```

### C2. 终端B：replay（推荐配置）

```bash
cd /home/user/wmd/lwmg
python scripts/replay_motion.py \
  --env-config configs/env/g1_squat_load.yaml \
  --tracker-config configs/tracker/sonic_dds_bridge.yaml \
  --reference-clip outputs/sonic_refs/clip_000 \
  --num-envs 1 \
  --max-steps 1000 \
  --step-mode physics \
  --strict-wait-dds \
  --headless \
  --diagnose \
  --diag-interval 50 \
  --trace-out outputs/replay_trace.npz
```

### C3. `replay_motion.py` 关键参数

1. `--step-mode physics|rl`
   - `physics`：每个 physics tick 发布/拉取 DDS，官方风格。
   - `rl`：每个 control step 发布/拉取一次，兼容旧链路。
2. `--strict-wait-dds`
   - 未收到首帧 DDS 指令前不推进仿真。
3. `--stream-reference`
   - 由 Python 将参考动作通过 ZMQManager pose topic 逐帧推给 C++。
4. `--report-rl-signals`
   - 只用于 physics 模式下额外输出 reward/done 诊断。

### C4. `sonic_dds_bridge.yaml` 关键模式

1. `dds.bridge_backend`
   - `proxy`（推荐）
   - `direct`
2. `dds.stale_policy`
   - `hold_last` / `default_pose` / `zeros`
3. `control_input.mode`
   - `none` / `zmq_manager`

## D. Python 内部冻结 SONIC 推理（无DDS）

```bash
cd /home/user/wmd/lwmg
python scripts/train_sonic_python.py \
  --env-config configs/env/g1_squat_load.yaml \
  --tracker-config configs/tracker/frozen_sonic_tracker.yaml \
  --reference-clip outputs/sonic_refs/clip_000 \
  --num-envs 64 \
  --max-steps 2000 \
  --headless \
  --diagnose \
  --diag-interval 100 \
  --trace-out outputs/train_sonic_python_trace.npz
```

说明：

1. 推理在 `SonicPythonRuntime` 内完成。
2. 动作语义：`q_target_abs = default_angles + raw_action * action_scale`。
3. `joint_order` 可选 `isaaclab/mujoco`，由配置决定是否做重排。

## E. 并行训练数据采集（推荐）

```bash
cd /home/user/wmd/lwmg
python scripts/collect_train_rollouts.py \
  --env-config configs/env/g1_squat_load.yaml \
  --tracker-config configs/tracker/frozen_sonic_tracker.yaml \
  --reference-clip outputs/sonic_refs/clip_000 \
  --num-envs 64 \
  --num-steps 4096 \
  --shard-steps 512 \
  --out-dir outputs/train_rollouts \
  --headless
```

可选扩展：

```bash
--include-sonic-obs
```

输出结构：

1. `meta.json`
2. `rollout_shard_*.npz`
3. 关键字段：`obs_*`, `next_obs_*`, `ref_*`, `sonic_raw_action_isaac`, `target_q_abs`, `reward`, `done`, `soft_failure`, `clamped_fraction`

## F. 对齐验证与误差比较

### F1. A/B 验证（Python vs C++）

```bash
cd /home/user/wmd/lwmg
python scripts/verify_sonic_alignment.py \
  --env-config configs/env/g1_squat_load.yaml \
  --dds-config configs/tracker/sonic_dds_bridge.yaml \
  --python-tracker-config configs/tracker/frozen_sonic_tracker.yaml \
  --reference-clip outputs/sonic_refs/clip_000 \
  --strict-wait-dds \
  --stream-reference \
  --max-steps 1000 \
  --trace-out outputs/verify_alignment_trace.npz \
  --headless
```

### F2. trace 对比（与官方CSV日志）

```bash
cd /home/user/wmd/lwmg
python scripts/compare_sonic_trace.py \
  --replay-trace outputs/replay_trace.npz \
  --sonic-log-dir /path/to/sonic_cpp_logs \
  --max-frames 1000 \
  --topk 8
```

## G. 你的上层模型训练入口

### G1. Flow

```bash
cd /home/user/wmd/lwmg
python scripts/train_flow_generator.py --config configs/train/train_flow_generator.yaml
```

### G2. World Model

```bash
cd /home/user/wmd/lwmg
python scripts/train_world_model.py --config configs/world_model/structured_closed_loop_wm.yaml --stage nominal
python scripts/train_world_model.py --config configs/world_model/structured_closed_loop_wm.yaml --stage residual
python scripts/train_world_model.py --config configs/world_model/structured_closed_loop_wm.yaml --stage joint
```

## H. 高频故障排查

1. 首帧 DDS 一直等不到：
   - C++是否真的在跑。
   - `--motion-data` 是否误传成 `clip_000`。
   - 5556 是否被旧进程占用。
2. `No module named isaaclab.app`：
   - 用 `isaaclab.sh -p` 启动脚本。
3. 机器人倒地/乱动：
   - 先跑 `verify_sonic_alignment.py`。
   - 再对比 `compare_sonic_trace.py` 的 `q/dq/action`。
