from __future__ import annotations

from isaaclab.assets import ArticulationCfg
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass

from .g1_robot_cfg import G1_JOINT_ORDER, make_g1_29dof_cfg


@configclass
class LwmgEnvCfg(DirectRLEnvCfg):
    # env
    decimation = 10
    episode_length_s = 10.0

    # spaces
    action_space = 29
    observation_space = 90
    state_space = 0

    # simulation
    sim: SimulationCfg = SimulationCfg(dt=0.002, render_interval=decimation)

    # robot
    robot_cfg: ArticulationCfg = make_g1_29dof_cfg(usd_path="/home/user/wmd/unitree_model/G1/29dof/usd/g1_29dof_rev_1_0/g1_29dof_rev_1_0.usd")

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=64, env_spacing=3.0, replicate_physics=True)

    # Internal env canonical order MUST stay in IsaacLab order.
    # MuJoCo ordering should only be handled by explicit remap tables.
    joint_names = list(G1_JOINT_ORDER)

    # action processing
    action_scale = 1.0
    action_clip = 0.35
    control_mode = "joint_target_delta"  # joint_target_delta | joint_target_absolute
    clamp_to_joint_limits = True

    # failure thresholds used for soft/hard flags in replay logging
    trunk_tilt_warn_rad = 0.55
    trunk_tilt_fail_rad = 1.05
    min_base_height = 0.35
