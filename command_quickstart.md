# G1 + Frozen SONIC 最短可跑清单

> 目标：最快完成一次“导出参考动作 -> C++推理 + IsaacLab replay”。

## 1) 进入环境

```bash
cd /home/user/wmd/lwmg
conda activate wmd_isaaclab
```

## 2) 导出参考动作

```bash
python scripts/export_sonic_references.py --config configs/sonic/sonic_export.yaml
ls -lah outputs/sonic_refs
```

## 3) 终端A启动官方 C++ SONIC（zmq_manager）

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

## 4) 终端B启动 IsaacLab replay

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
  --headless
```

## 5) 可选：保存追踪诊断

```bash
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

## 6) 结束顺序

1. 先停终端B（replay）。
2. 再停终端A（C++ deploy）。

## 常见坑（最关键三条）

1. `--motion-data` 要传目录根（`outputs/sonic_refs`），不是 `clip_000`。
2. 如果报 `No module named isaaclab.app`，请用 IsaacLab 启动器：

```bash
cd /home/user/wmd/IsaacLab
./isaaclab.sh -p /home/user/wmd/lwmg/scripts/replay_motion.py --help
```

3. 如果卡在等待 DDS，检查 C++ 是否正常启动、端口 5556 是否被旧进程占用。
