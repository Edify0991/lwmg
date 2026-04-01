from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# SONIC policy_parameters.hpp (IsaacLab order)
G1_JOINT_ORDER = [
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

# SONIC gatherer indices (from policy_parameters.hpp)
LOWER_BODY_JOINT_INDICES_MUJOCO_ORDER_IN_ISAACLAB_INDEX = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18]
WRIST_JOINT_INDICES_ISAACLAB_ORDER_IN_ISAACLAB_INDEX = [23, 24, 25, 26, 27, 28]
ISAACLAB_TO_MUJOCO = [0, 3, 6, 9, 13, 17, 1, 4, 7, 10, 14, 18, 2, 5, 8, 11, 15, 19, 21, 23, 25, 27, 12, 16, 20, 22, 24, 26, 28]
MUJOCO_TO_ISAACLAB = [0, 6, 12, 1, 7, 13, 2, 8, 14, 3, 9, 15, 22, 4, 10, 16, 23, 5, 11, 17, 24, 18, 25, 19, 26, 20, 27, 21, 28]

# SONIC policy_parameters.hpp constants used for action target reconstruction:
# target = action * g1_action_scale + default_angles
_NATURAL_FREQ = 10.0 * 2.0 * 3.1415926535
_ARMATURE_5020 = 0.003609725
_ARMATURE_7520_14 = 0.010177520
_ARMATURE_7520_22 = 0.025101925
_ARMATURE_4010 = 0.00425

_STIFFNESS_5020 = _ARMATURE_5020 * _NATURAL_FREQ * _NATURAL_FREQ
_STIFFNESS_7520_14 = _ARMATURE_7520_14 * _NATURAL_FREQ * _NATURAL_FREQ
_STIFFNESS_7520_22 = _ARMATURE_7520_22 * _NATURAL_FREQ * _NATURAL_FREQ
_STIFFNESS_4010 = _ARMATURE_4010 * _NATURAL_FREQ * _NATURAL_FREQ

_DAMPING_RATIO = 2.0
_DAMPING_5020 = 2.0 * _DAMPING_RATIO * _ARMATURE_5020 * _NATURAL_FREQ
_DAMPING_7520_14 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_14 * _NATURAL_FREQ
_DAMPING_7520_22 = 2.0 * _DAMPING_RATIO * _ARMATURE_7520_22 * _NATURAL_FREQ
_DAMPING_4010 = 2.0 * _DAMPING_RATIO * _ARMATURE_4010 * _NATURAL_FREQ

_ACTION_SCALE_5020 = 0.25 * 25.0 / _STIFFNESS_5020
_ACTION_SCALE_7520_14 = 0.25 * 88.0 / _STIFFNESS_7520_14
_ACTION_SCALE_7520_22 = 0.25 * 139.0 / _STIFFNESS_7520_22
_ACTION_SCALE_4010 = 0.25 * 5.0 / _STIFFNESS_4010

G1_SONIC_ACTION_SCALE = [
    _ACTION_SCALE_7520_22,  # left_hip_pitch_joint
    _ACTION_SCALE_7520_22,  # left_hip_roll_joint
    _ACTION_SCALE_7520_14,  # left_hip_yaw_joint
    _ACTION_SCALE_7520_22,  # left_knee_joint
    _ACTION_SCALE_5020,  # left_ankle_pitch_joint
    _ACTION_SCALE_5020,  # left_ankle_roll_joint
    _ACTION_SCALE_7520_22,  # right_hip_pitch_joint
    _ACTION_SCALE_7520_22,  # right_hip_roll_joint
    _ACTION_SCALE_7520_14,  # right_hip_yaw_joint
    _ACTION_SCALE_7520_22,  # right_knee_joint
    _ACTION_SCALE_5020,  # right_ankle_pitch_joint
    _ACTION_SCALE_5020,  # right_ankle_roll_joint
    _ACTION_SCALE_7520_14,  # waist_yaw_joint
    _ACTION_SCALE_5020,  # waist_roll_joint
    _ACTION_SCALE_5020,  # waist_pitch_joint
    _ACTION_SCALE_5020,  # left_shoulder_pitch_joint
    _ACTION_SCALE_5020,  # left_shoulder_roll_joint
    _ACTION_SCALE_5020,  # left_shoulder_yaw_joint
    _ACTION_SCALE_5020,  # left_elbow_joint
    _ACTION_SCALE_5020,  # left_wrist_roll_joint
    _ACTION_SCALE_4010,  # left_wrist_pitch_joint
    _ACTION_SCALE_4010,  # left_wrist_yaw_joint
    _ACTION_SCALE_5020,  # right_shoulder_pitch_joint
    _ACTION_SCALE_5020,  # right_shoulder_roll_joint
    _ACTION_SCALE_5020,  # right_shoulder_yaw_joint
    _ACTION_SCALE_5020,  # right_elbow_joint
    _ACTION_SCALE_5020,  # right_wrist_roll_joint
    _ACTION_SCALE_4010,  # right_wrist_pitch_joint
    _ACTION_SCALE_4010,  # right_wrist_yaw_joint
]

# SONIC default standing pose (policy_parameters.hpp::default_angles)
G1_DEFAULT_JOINT_ANGLES = [
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    -0.312,
    0.0,
    0.0,
    0.669,
    -0.363,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
    0.2,
    -0.2,
    0.0,
    0.6,
    0.0,
    0.0,
    0.0,
]


def _default_joint_pos_map() -> dict[str, float]:
    return {name: angle for name, angle in zip(G1_JOINT_ORDER, G1_DEFAULT_JOINT_ANGLES, strict=True)}


def make_g1_29dof_cfg(
    prim_path: str = "/World/envs/env_.*/Robot",
    usd_path: str | None = None,
    base_height: float = 0.75,
    enable_self_collisions: bool = False,
) -> ArticulationCfg:
    usd_path = usd_path or f"{ISAAC_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd"

    return ArticulationCfg(
        prim_path=prim_path,
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.0,
                angular_damping=0.0,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=enable_self_collisions,
                fix_root_link=False,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, base_height),
            rot=(1.0, 0.0, 0.0, 0.0),
            joint_pos=_default_joint_pos_map(),
            joint_vel={".*": 0.0},
        ),
        # Keep full nominal joint range to match SONIC target semantics.
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[
                    ".*_hip_yaw_joint",
                    ".*_hip_roll_joint",
                    ".*_hip_pitch_joint",
                    ".*_knee_joint",
                ],
                effort_limit={
                    ".*_hip_yaw_joint": 88.0,
                    ".*_hip_roll_joint": 139.0,
                    ".*_hip_pitch_joint": 139.0,
                    ".*_knee_joint": 139.0,
                },
                velocity_limit={
                    ".*_hip_yaw_joint": 32.0,
                    ".*_hip_roll_joint": 32.0,
                    ".*_hip_pitch_joint": 32.0,
                    ".*_knee_joint": 20.0,
                },
                stiffness={
                    ".*_hip_yaw_joint": _STIFFNESS_7520_14,
                    ".*_hip_roll_joint": _STIFFNESS_7520_22,
                    ".*_hip_pitch_joint": _STIFFNESS_7520_22,
                    ".*_knee_joint": _STIFFNESS_7520_22,
                },
                damping={
                    ".*_hip_yaw_joint": _DAMPING_7520_14,
                    ".*_hip_roll_joint": _DAMPING_7520_22,
                    ".*_hip_pitch_joint": _DAMPING_7520_22,
                    ".*_knee_joint": _DAMPING_7520_22,
                },
                armature={
                    ".*_hip_.*": 0.03,
                    ".*_knee_joint": 0.03,
                },
                saturation_effort=180.0,
            ),
            "feet": DCMotorCfg(
                joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
                stiffness={
                    ".*_ankle_pitch_joint": 2.0 * _STIFFNESS_5020,
                    ".*_ankle_roll_joint": 2.0 * _STIFFNESS_5020,
                },
                damping={
                    ".*_ankle_pitch_joint": 2.0 * _DAMPING_5020,
                    ".*_ankle_roll_joint": 2.0 * _DAMPING_5020,
                },
                effort_limit={
                    ".*_ankle_pitch_joint": 50.0,
                    ".*_ankle_roll_joint": 50.0,
                },
                velocity_limit={
                    ".*_ankle_pitch_joint": 37.0,
                    ".*_ankle_roll_joint": 37.0,
                },
                armature=0.03,
                saturation_effort=80.0,
            ),
            "waist": ImplicitActuatorCfg(
                joint_names_expr=["waist_.*_joint"],
                effort_limit={
                    "waist_yaw_joint": 88.0,
                    "waist_roll_joint": 50.0,
                    "waist_pitch_joint": 50.0,
                },
                velocity_limit={
                    "waist_yaw_joint": 32.0,
                    "waist_roll_joint": 37.0,
                    "waist_pitch_joint": 37.0,
                },
                stiffness={
                    "waist_yaw_joint": _STIFFNESS_7520_14,
                    "waist_roll_joint": 2.0 * _STIFFNESS_5020,
                    "waist_pitch_joint": 2.0 * _STIFFNESS_5020,
                },
                damping={
                    "waist_yaw_joint": _DAMPING_7520_14,
                    "waist_roll_joint": 2.0 * _DAMPING_5020,
                    "waist_pitch_joint": 2.0 * _DAMPING_5020,
                },
                armature=0.001,
            ),
            "arms": ImplicitActuatorCfg(
                joint_names_expr=[
                    ".*_shoulder_pitch_joint",
                    ".*_shoulder_roll_joint",
                    ".*_shoulder_yaw_joint",
                    ".*_elbow_joint",
                    ".*_wrist_roll_joint",
                    ".*_wrist_pitch_joint",
                    ".*_wrist_yaw_joint",
                ],
                effort_limit={
                    ".*_shoulder_pitch_joint": 25.0,
                    ".*_shoulder_roll_joint": 25.0,
                    ".*_shoulder_yaw_joint": 25.0,
                    ".*_elbow_joint": 25.0,
                    ".*_wrist_roll_joint": 25.0,
                    ".*_wrist_pitch_joint": 5.0,
                    ".*_wrist_yaw_joint": 5.0,
                },
                velocity_limit=100.0,
                stiffness={
                    ".*_shoulder_pitch_joint": _STIFFNESS_5020,
                    ".*_shoulder_roll_joint": _STIFFNESS_5020,
                    ".*_shoulder_yaw_joint": _STIFFNESS_5020,
                    ".*_elbow_joint": _STIFFNESS_5020,
                    ".*_wrist_roll_joint": _STIFFNESS_5020,
                    ".*_wrist_pitch_joint": _STIFFNESS_4010,
                    ".*_wrist_yaw_joint": _STIFFNESS_4010,
                },
                damping={
                    ".*_shoulder_pitch_joint": _DAMPING_5020,
                    ".*_shoulder_roll_joint": _DAMPING_5020,
                    ".*_shoulder_yaw_joint": _DAMPING_5020,
                    ".*_elbow_joint": _DAMPING_5020,
                    ".*_wrist_roll_joint": _DAMPING_5020,
                    ".*_wrist_pitch_joint": _DAMPING_4010,
                    ".*_wrist_yaw_joint": _DAMPING_4010,
                },
                armature=0.001,
            ),
        },
    )
