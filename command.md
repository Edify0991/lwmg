# LWMG G1 + Frozen SONIC 运行手册（当前仓库代码对齐版）

更新时间：2026-03-27

## 1. 总体分层与模式

当前代码已经形成两条主链路：

1. `官方 C++ SONIC 推理 + IsaacLab 物理`（推荐用于 sim2sim / 真实部署前对齐）
2. `Python 内部 SONIC ONNX 推理 + IsaacLab 物理`（推荐用于并行数据采集与训练）

分层可理解为：

- 回放阶段（sim2sim）：
  1. 高层：参考动作来源（C++内部motion reader 或 Python流式发送）
  2. 中层：冻结 SONIC 策略推理（C++）
  3. 底层：IsaacLab 物理环境（Python）
- 训练阶段（建议）：
  1. 高层：你自己的生成模型/世界模型
  2. 中层：冻结 SONIC ONNX（Python in-process）
  3. 底层：IsaacLab 并行环境

---

## 2. 一次性准备

### 2.1 进入仓库

```bash
cd /home/user/wmd/lwmg
```

### 2.2 激活环境

```bash
conda activate wmd_isaaclab
```

### 2.3 快速检查关键文件存在

```bash
ls -l \
  configs/env/g1_squat_load.yaml \
  configs/tracker/sonic_dds_bridge.yaml \
  configs/tracker/frozen_sonic_tracker.yaml \
  configs/sonic/sonic_export.yaml
```

### 2.4 如果 `python scripts/...` 提示找不到 `isaaclab.app`

使用 IsaacLab 启动器执行脚本：

```bash
cd /home/user/wmd/IsaacLab
./isaaclab.sh -p /home/user/wmd/lwmg/scripts/replay_motion.py --help
```

---

## 3. 参考动作导出（支持多数据源）

脚本：`scripts/export_sonic_references.py`
配置：`configs/sonic/sonic_export.yaml`

### 3.1 支持的数据源类型（`source.type`）

1. `lafan1_retarget_csv`
2. `amass_retarget_csv`
3. `generic_csv`
4. `generated_npz`
5. `generated_pt`
6. `synthetic_zero`（仅调试，不建议用于真实验证）

### 3.2 导出命令

```bash
cd /home/user/wmd/lwmg
python scripts/export_sonic_references.py --config configs/sonic/sonic_export.yaml
```

### 3.3 导出结果检查

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

---

## 4. 回放链路 A：官方 C++ 推理 + IsaacLab（DDS 桥接，推荐）

脚本：`scripts/replay_motion.py`
配置：`configs/tracker/sonic_dds_bridge.yaml`

> 这条链路最接近官方部署路径：推理由 C++ 完成，Python/IsaacLab 负责仿真与状态交换。

### 4.1 终端 A：启动官方 C++ SONIC

推荐使用你仓库内 thirdparty 的 deploy：

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

说明：

1. `--motion-data` 传“目录根路径”（例如 `outputs/sonic_refs`），不是 `clip_000`。
2. `--input-type zmq_manager` 对应你现在的“无 keyboard 自动启动/流式接入”模式。

### 4.2 终端 B：启动 IsaacLab replay

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

### 4.3 `replay_motion.py` 关键模式解释

#### 4.3.1 `--step-mode`

1. `physics`（默认，推荐）
   - 每个 physics tick 都发布 lowstate、拉取 DDS 命令、再执行 `step_physics`。
   - 更接近官方 MuJoCo 侧“高频交互”。
2. `rl`
   - 每个 control step 执行一次发布/拉取，再走 `env.step()`。
   - 兼容旧逻辑，诊断时可用。

#### 4.3.2 `--strict-wait-dds`

1. 打开后：在首次收到有效 DDS 命令前，不推进仿真。
2. 配套参数：
   - `--wait-sleep-s`
   - `--strict-wait-timeout-s`

#### 4.3.3 `--stream-reference`

1. 关闭（默认）：C++ 使用自己的 motion reader（读取 `--motion-data`）。
2. 打开：Python 将 `--reference-clip` 逐帧通过 ZMQManager `pose` topic 推给 C++。
3. 要求：`control_input.mode=zmq_manager` 且 `planner_mode=false`。

#### 4.3.4 `--report-rl-signals`

仅在 `step-mode=physics` 下额外计算 reward/done 诊断量，不改变 C++ 控制闭环。

### 4.4 DDS 配置模式解释（`sonic_dds_bridge.yaml`）

#### 4.4.1 `dds.bridge_backend`

1. `proxy`（推荐）
   - 在独立子进程跑官方 bridge，兼容性更稳。
2. `direct`
   - IsaacLab 进程内直接 import Unitree Python SDK，对 Python/CycloneDDS ABI 更敏感。

#### 4.4.2 `dds.stale_policy`

1. `hold_last`：命令断流时保持上一帧目标。
2. `default_pose`：回默认姿态。
3. `zeros`：全零目标。

#### 4.4.3 `control_input.mode`

1. `none`：不发送自动 start/stop。
2. `zmq_manager`：自动发送 command（start/stop）并支持 pose 流式输入。

---

## 5. 回放链路 B：Python 内部冻结 SONIC 推理（无 DDS）

脚本：`scripts/train_sonic_python.py`
配置：`configs/tracker/frozen_sonic_tracker.yaml`

用途：

1. 快速验证 in-process 推理逻辑。
2. 训练前 sanity-check。
3. 不依赖 C++ 进程，调试更快。

命令：

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

模式说明：

1. 推理发生在 Python `SonicPythonRuntime` 内。
2. 动作语义：`q_target_abs = default_angles + raw_action * action_scale`。
3. `joint_order` 支持 `isaaclab/mujoco` 映射处理（配置在 `frozen_sonic_tracker.yaml`）。

---

## 6. 训练数据采集（并行，推荐）

脚本：`scripts/collect_train_rollouts.py`

用途：

1. 并行采集训练集。
2. 适配“冻结 SONIC + 生成模型/世界模型”后续训练。

命令：

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

可选：保存 SONIC 观测向量（文件更大）：

```bash
--include-sonic-obs
```

输出说明：

1. 目录：`outputs/train_rollouts/sonic_rollout_YYYYMMDD_HHMMSS/`
2. 文件：
   - `meta.json`
   - `rollout_shard_00000.npz` 等
3. shard 里包含：
   - `obs_*` / `next_obs_*`
   - `ref_*`
   - `sonic_raw_action_isaac`
   - `target_q_abs`
   - `reward`, `done`, `soft_failure`, `clamped_fraction`
   - 可选 `sonic_decoder_obs`, `sonic_encoder_obs`

---

## 7. 对齐验证与误差分析

### 7.1 A/B 对齐（Python ONNX vs C++ DDS）

脚本：`scripts/verify_sonic_alignment.py`

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

### 7.2 Trace 与官方日志对比

脚本：`scripts/compare_sonic_trace.py`

```bash
cd /home/user/wmd/lwmg
python scripts/compare_sonic_trace.py \
  --replay-trace outputs/replay_trace.npz \
  --sonic-log-dir /path/to/sonic_cpp_logs \
  --max-frames 1000 \
  --topk 8
```

---

## 8. 训练脚本（你后续模型层）

### 8.1 Flow 生成模型

脚本：`scripts/train_flow_generator.py`

```bash
cd /home/user/wmd/lwmg
python scripts/train_flow_generator.py --config configs/train/train_flow_generator.yaml
```

模式：`flow_family = flow_matching | rectified_flow | mean_flow`

### 8.2 世界模型

脚本：`scripts/train_world_model.py`

```bash
cd /home/user/wmd/lwmg
python scripts/train_world_model.py --config configs/world_model/structured_closed_loop_wm.yaml --stage nominal
python scripts/train_world_model.py --config configs/world_model/structured_closed_loop_wm.yaml --stage residual
python scripts/train_world_model.py --config configs/world_model/structured_closed_loop_wm.yaml --stage joint
```

`--stage` 模式：

1. `nominal`
2. `residual`
3. `joint`

---

## 9. 生成动作接入建议（与你的后续 pipeline 对齐）

1. 生成模型输出为 `npz` 或 `pt`。
2. 在 `configs/sonic/sonic_export.yaml` 切到 `source.type=generated_npz` 或 `generated_pt`，并配置 `keys`。
3. 重新运行导出脚本，得到标准 SONIC clip。
4. 回放阶段可选：
   - 让 C++ 直接读 `--motion-data`。
   - 或 replay 里开启 `--stream-reference`，由 Python 逐帧喂给 C++。

---

## 10. 常见问题速查

1. 现象：`Timed out while waiting for first DDS command`
   - 检查 C++ 是否正常运行。
   - 检查 `--motion-data` 是否为根目录而非 `clip_000`。
   - 检查 5556 端口是否被旧进程占用。
2. 现象：`No module named isaaclab.app`
   - 改用 `isaaclab.sh -p` 启动脚本。
3. 现象：机器人乱动/倒地
   - 先做 `verify_sonic_alignment.py`，确认 Python 与 C++ 目标输出是否对齐。
   - 再用 `compare_sonic_trace.py` 对照 `q/dq/action` 差异。

