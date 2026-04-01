from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import isaaclab.sim as sim_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane

from lwmg.envs.observations import Observation

from .g1_robot_cfg import G1_DEFAULT_JOINT_ANGLES, G1_JOINT_ORDER, ISAACLAB_TO_MUJOCO, MUJOCO_TO_ISAACLAB
from .lwmg_env_cfg import LwmgEnvCfg


class LwmgEnv(DirectRLEnv):
    cfg: LwmgEnvCfg

    def __init__(self, cfg: LwmgEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        if list(self.cfg.joint_names) != list(G1_JOINT_ORDER):
            raise ValueError(
                "cfg.joint_names must be IsaacLab order (G1_JOINT_ORDER). "
                "Do not pass MuJoCo-order joint_names into LwmgEnv, otherwise remapping will be incorrect."
            )

        self._joint_ids = self._resolve_joint_ids(self.cfg.joint_names)
        self.num_joints = len(self._joint_ids)

        self.actions = torch.zeros(self.num_envs, self.num_joints, device=self.device)
        self.prev_action = torch.zeros_like(self.actions)
        self.tracker_prev_action = torch.zeros_like(self.actions)
        self._last_target = torch.zeros_like(self.actions)
        self._soft_failure = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._last_clamped_fraction = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._default_joint_pos = torch.tensor(G1_DEFAULT_JOINT_ANGLES, dtype=torch.float32, device=self.device)
        if self._default_joint_pos.numel() != self.num_joints:
            raise ValueError(
                f"Default joint pose dim={self._default_joint_pos.numel()} does not match active SONIC joint dim={self.num_joints}"
            )
        if len(ISAACLAB_TO_MUJOCO) != self.num_joints or len(MUJOCO_TO_ISAACLAB) != self.num_joints:
            raise ValueError(
                "Joint remapping table size mismatch. "
                f"isaaclab_to_mujoco={len(ISAACLAB_TO_MUJOCO)} mujoco_to_isaaclab={len(MUJOCO_TO_ISAACLAB)} "
                f"num_joints={self.num_joints}"
            )

        self._isaac_to_mujoco_idx = torch.tensor(ISAACLAB_TO_MUJOCO, dtype=torch.long, device=self.device)
        self._mujoco_to_isaac_idx = torch.tensor(MUJOCO_TO_ISAACLAB, dtype=torch.long, device=self.device)
        self._control_dt = float(self.cfg.sim.dt) * float(self.cfg.decimation)
        self._prev_root_lin_vel_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)

        self.joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_ids]

    def _resolve_joint_ids(self, expected_names: list[str]) -> list[int]:
        name_to_idx = {name: i for i, name in enumerate(self.robot.joint_names)}
        missing = [name for name in expected_names if name not in name_to_idx]
        if missing:
            raise RuntimeError(f"Missing joints in articulation for SONIC alignment: {missing}")
        return [name_to_idx[name] for name in expected_names]

    def _setup_scene(self):
        self.robot = Articulation(self.cfg.robot_cfg)
        spawn_ground_plane(prim_path="/World/ground", cfg=GroundPlaneCfg())
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if actions.shape[-1] != self.num_joints:
            raise ValueError(f"Expected action dim={self.num_joints}, got {actions.shape[-1]}")

        if self.cfg.control_mode == "joint_target_absolute":
            self.actions = actions.clone()
            return

        clipped = torch.clamp(actions, -float(self.cfg.action_clip), float(self.cfg.action_clip))
        self.actions = clipped.clone()

    def _apply_action(self) -> None:
        current = self.robot.data.joint_pos[:, self._joint_ids]
        if self.cfg.control_mode == "joint_target_absolute":
            target_raw = self.actions * float(self.cfg.action_scale)
        else:
            target_raw = current + self.actions * float(self.cfg.action_scale)

        if bool(getattr(self.cfg, "clamp_to_joint_limits", True)):
            limits = self.robot.data.soft_joint_pos_limits[:, self._joint_ids]
            target = torch.clamp(target_raw, min=limits[..., 0], max=limits[..., 1])

            clamped = (target - target_raw).abs() > 1e-6
            self._last_clamped_fraction = clamped.to(dtype=torch.float32).mean(dim=-1)
        else:
            target = target_raw
            self._last_clamped_fraction.zero_()

        self.robot.set_joint_position_target(target, joint_ids=self._joint_ids)
        self.prev_action = self.actions.clone()
        self._last_target = target.clone()

    def step_physics(self, actions: torch.Tensor) -> None:
        """Advance one physics tick without RL reward/reset bookkeeping."""
        action = actions.to(self.device)
        self._pre_physics_step(action)

        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors()
        self._sim_step_counter += 1

        self._apply_action()
        self.scene.write_data_to_sim()
        self.sim.step(render=False)
        if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
            self.sim.render()
        self.scene.update(dt=self.physics_dt)

    def get_done_flags(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (terminated, timeout) without auto-reset."""
        return self._get_dones()

    def get_rl_signals(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (reward, terminated, timeout) without stepping/resetting."""
        terminated, timeout = self._get_dones()
        reward = self._get_rewards()
        return reward, terminated, timeout

    def get_done_diagnostics(self) -> dict[str, torch.Tensor]:
        """Return done flags and underlying scalar signals for debugging/reset policy."""
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_ids]

        gravity_b = self.robot.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity_b[:, 2], min=-1.0, max=1.0))

        base_height = self.robot.data.root_pos_w[:, 2]
        base_height_fail = base_height < float(self.cfg.min_base_height)
        tilt_fail = tilt > float(self.cfg.trunk_tilt_fail_rad)
        numeric_fail = torch.isnan(self.joint_pos).any(dim=-1) | torch.isnan(self.joint_vel).any(dim=-1)
        timeout = self.episode_length_buf >= self.max_episode_length - 1

        terminated = base_height_fail | tilt_fail | numeric_fail
        done = terminated | timeout

        return {
            "base_height": base_height.detach().clone(),
            "tilt": tilt.detach().clone(),
            "base_height_fail": base_height_fail.detach().clone(),
            "tilt_fail": tilt_fail.detach().clone(),
            "numeric_fail": numeric_fail.detach().clone(),
            "timeout": timeout.detach().clone(),
            "terminated": terminated.detach().clone(),
            "done": done.detach().clone(),
        }

    def _get_observations(self) -> dict:
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_ids]
        base_ang_vel = self.robot.data.root_ang_vel_b

        obs = torch.cat((self.joint_pos, self.joint_vel, self.prev_action, base_ang_vel), dim=-1)
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        tracking_error = torch.mean(torch.abs(self._last_target - self.joint_pos), dim=-1)

        gravity_b = self.robot.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity_b[:, 2], min=-1.0, max=1.0))
        self._soft_failure = tilt > float(self.cfg.trunk_tilt_warn_rad)

        base_height = self.robot.data.root_pos_w[:, 2]
        alive = torch.ones_like(base_height)
        reward = alive - 0.5 * tracking_error - 0.1 * tilt
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        self.joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        self.joint_vel = self.robot.data.joint_vel[:, self._joint_ids]

        gravity_b = self.robot.data.projected_gravity_b
        tilt = torch.acos(torch.clamp(-gravity_b[:, 2], min=-1.0, max=1.0))

        base_height_fail = self.robot.data.root_pos_w[:, 2] < float(self.cfg.min_base_height)
        tilt_fail = tilt > float(self.cfg.trunk_tilt_fail_rad)
        numeric_fail = torch.isnan(self.joint_pos).any(dim=-1) | torch.isnan(self.joint_vel).any(dim=-1)

        terminated = base_height_fail | tilt_fail | numeric_fail
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        return terminated, time_out

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.robot._ALL_INDICES
        super()._reset_idx(env_ids)

        default_root_state = self.robot.data.default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        joint_vel = self.robot.data.default_joint_vel[env_ids].clone()

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self.prev_action[env_ids] = 0.0
        self.tracker_prev_action[env_ids] = 0.0
        self.actions[env_ids] = 0.0
        self._last_target[env_ids] = joint_pos[:, self._joint_ids]
        self._soft_failure[env_ids] = False
        self._last_clamped_fraction[env_ids] = 0.0
        self._prev_root_lin_vel_w[env_ids] = self.robot.data.root_lin_vel_w[env_ids]

    def get_tracker_observation(self, env_id: int) -> Observation:
        q_abs = self.robot.data.joint_pos[env_id, self._joint_ids].detach().clone()
        # Official SONIC deployment logs body_q as (q - default_angles).
        q = q_abs - self._default_joint_pos
        dq = self.robot.data.joint_vel[env_id, self._joint_ids].detach().clone()
        imu_gyro = self.robot.data.root_ang_vel_b[env_id].detach().clone()
        gravity_dir = self.robot.data.projected_gravity_b[env_id].detach().clone()
        root_quat = self.robot.data.root_quat_w[env_id].detach().clone()

        tracking_error = torch.mean(torch.abs(self._last_target[env_id] - q_abs)).unsqueeze(0)

        return Observation(
            q=q,
            dq=dq,
            imu_accel=torch.zeros(3, dtype=q.dtype, device=q.device),
            imu_gyro=imu_gyro,
            prev_action=self.tracker_prev_action[env_id].detach().clone(),
            contacts=torch.ones(2, dtype=q.dtype, device=q.device),
            tracking_error_summary=tracking_error,
            root_quat_wxyz=root_quat,
            projected_gravity=gravity_dir,
        )

    def get_tracker_observations(self) -> list[Observation]:
        return [self.get_tracker_observation(i) for i in range(self.num_envs)]

    def set_tracker_prev_action(self, raw_actions: torch.Tensor) -> None:
        if raw_actions.shape != self.tracker_prev_action.shape:
            raise ValueError(
                f"Expected raw_actions shape {tuple(self.tracker_prev_action.shape)}, got {tuple(raw_actions.shape)}"
            )
        self.tracker_prev_action = raw_actions.detach().to(self.device).clone()

    def get_joint_state_abs_vel(self, env_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        q_abs = self.robot.data.joint_pos[env_id, self._joint_ids].detach().clone()
        dq = self.robot.data.joint_vel[env_id, self._joint_ids].detach().clone()
        return q_abs, dq

    def body_target_mujoco_to_isaaclab(self, body_target_mujoco: np.ndarray) -> torch.Tensor:
        if body_target_mujoco.ndim != 1:
            raise ValueError(f"Expected rank-1 mujoco target, got shape={body_target_mujoco.shape}")
        if body_target_mujoco.shape[0] != self.num_joints:
            raise ValueError(
                f"Expected mujoco target dim={self.num_joints}, got {body_target_mujoco.shape[0]}"
            )
        target_mj = torch.tensor(body_target_mujoco, dtype=torch.float32, device=self.device)
        return target_mj[self._mujoco_to_isaac_idx]

    def build_unitree_lowstate_payload(
        self,
        env_id: int,
        sim_time_s: float,
        dt_s: float | None = None,
    ) -> dict[str, np.ndarray | float]:
        q_abs_isaac = self.robot.data.joint_pos[env_id, self._joint_ids].detach()
        dq_isaac = self.robot.data.joint_vel[env_id, self._joint_ids].detach()
        ddq_isaac = self.robot.data.joint_acc[env_id, self._joint_ids].detach()

        applied_torque = getattr(self.robot.data, "applied_torque", None)
        if applied_torque is None:
            tau_isaac = torch.zeros_like(q_abs_isaac)
        else:
            tau_isaac = applied_torque[env_id, self._joint_ids].detach()

        q_abs_mj = q_abs_isaac[self._isaac_to_mujoco_idx].to(dtype=torch.float32).cpu().numpy()
        dq_mj = dq_isaac[self._isaac_to_mujoco_idx].to(dtype=torch.float32).cpu().numpy()
        ddq_mj = ddq_isaac[self._isaac_to_mujoco_idx].to(dtype=torch.float32).cpu().numpy()
        tau_mj = tau_isaac[self._isaac_to_mujoco_idx].to(dtype=torch.float32).cpu().numpy()

        root_pos_w = self.robot.data.root_pos_w[env_id].detach().to(dtype=torch.float32)
        root_quat_wxyz = self.robot.data.root_quat_w[env_id].detach().to(dtype=torch.float32)
        root_lin_vel_w = self.robot.data.root_lin_vel_w[env_id].detach().to(dtype=torch.float32)
        root_lin_vel_b = self.robot.data.root_lin_vel_b[env_id].detach().to(dtype=torch.float32)
        root_ang_vel_b = self.robot.data.root_ang_vel_b[env_id].detach().to(dtype=torch.float32)

        sample_dt = float(self._control_dt if dt_s is None else dt_s)
        prev_lin_vel_w = self._prev_root_lin_vel_w[env_id]
        lin_acc_w = (root_lin_vel_w - prev_lin_vel_w) / max(sample_dt, 1.0e-6)
        self._prev_root_lin_vel_w[env_id] = root_lin_vel_w

        floating_base_pose = torch.cat((root_pos_w, root_quat_wxyz), dim=0).cpu().numpy()
        floating_base_vel = torch.cat((root_lin_vel_w, root_ang_vel_b), dim=0).cpu().numpy()
        floating_base_acc = torch.cat((lin_acc_w, torch.zeros(3, device=self.device)), dim=0).cpu().numpy()
        secondary_imu_vel = torch.cat((root_lin_vel_b, root_ang_vel_b), dim=0).cpu().numpy()

        return {
            "floating_base_pose": floating_base_pose.astype(np.float32, copy=False),
            "floating_base_vel": floating_base_vel.astype(np.float32, copy=False),
            "floating_base_acc": floating_base_acc.astype(np.float32, copy=False),
            "secondary_imu_quat": root_quat_wxyz.cpu().numpy().astype(np.float32, copy=False),
            "secondary_imu_vel": secondary_imu_vel.astype(np.float32, copy=False),
            "body_q": q_abs_mj.astype(np.float32, copy=False),
            "body_dq": dq_mj.astype(np.float32, copy=False),
            "body_ddq": ddq_mj.astype(np.float32, copy=False),
            "body_tau_est": tau_mj.astype(np.float32, copy=False),
            "left_hand_q": np.zeros((0,), dtype=np.float32),
            "left_hand_dq": np.zeros((0,), dtype=np.float32),
            "right_hand_q": np.zeros((0,), dtype=np.float32),
            "right_hand_dq": np.zeros((0,), dtype=np.float32),
            "time": float(sim_time_s),
        }

    @property
    def soft_failure(self) -> torch.Tensor:
        return self._soft_failure

    @property
    def last_clamped_fraction(self) -> torch.Tensor:
        return self._last_clamped_fraction
