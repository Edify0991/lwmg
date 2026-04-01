顺序对照（最关键）

你这边按 YAML 顺序读取 decoder/encoder 名称
sonic_obs_builder.py (line 113)
sonic_obs_builder.py (line 121)

你这边按该顺序拼接 decoder/encoder
sonic_obs_builder.py (line 277)
sonic_obs_builder.py (line 290)

官方同样按配置顺序构建 active observation 列表
g1_deploy_onnx_ref.cpp (line 1658)
g1_deploy_onnx_ref.cpp (line 1701)

你当前启用项顺序（官方 YAML）
observation_config.yaml (line 5)
observation_config.yaml (line 28)

维度对照

官方 registry 维度定义
g1_deploy_onnx_ref.cpp (line 1574)

你这边 registry 维度定义
sonic_obs_builder.py (line 335)

我做了自动比对：

his_*, motion_*, smpl_*, vr_* 这些启用项维度全部一致。
token_state 官方在 C++ 是动态 token_dim（不是写死数字），你这边按 encoder_dim（当前 64）处理，效果一致。
decoder 总维：994（官方 registry 数值项相加是 930，加上 token 64 即 994）
encoder 总维：1762（一致）
0 padding 对照

官方 policy obs 先全 0
g1_deploy_onnx_ref.cpp (line 1834)

官方 encoder obs 先全 0，非 required 项保持 0
g1_deploy_onnx_ref.cpp (line 1861)
g1_deploy_onnx_ref.cpp (line 1942)

官方历史 GetLatest 不足时补零 entry
state_logger.cpp (line 183)
state_logger.cpp (line 462)

你这边 encoder 非 required 项补零
sonic_obs_builder.py (line 291)

你这边历史不足补零 observation，并保持 oldest->newest 排列
sonic_obs_builder.py (line 640)
sonic_obs_builder.py (line 648)

你这边 token_state 先占位，再在推理前写入 encoder token
sonic_obs_builder.py (line 455)
sonic_onnx_runner.py (line 168)

一个你应注意的小点

observation_config.yaml 顶部注释写的 “Total dimension: 436” 与当前启用项不一致（当前按配置实际是 decoder 994）。这行注释看起来是旧值。
observation_config.yaml (line 3)