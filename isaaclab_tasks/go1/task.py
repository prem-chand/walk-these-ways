from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import importlib.util

if importlib.util.find_spec("torch"):
    import torch
else:
    torch = None

if TYPE_CHECKING:
    import torch as torch_typing

from isaaclab_tasks.go1.config import Go1TaskCfg


@dataclass
class Go1LocomotionTask:
    """IsaacLab-ready task scaffold for Go1 locomotion.

    This scaffold isolates configuration validation and exposes the
    computed observation dimensions used by training code.
    """

    cfg: Go1TaskCfg

    def __post_init__(self) -> None:
        self.cfg.validate()
        if torch is None:
            self.obs_history_dim = self.cfg.env.num_observations * self.cfg.env.num_observation_history
            self.privileged_obs_dim = self.cfg.env.num_privileged_obs
            return
        self.device = torch.device("cpu")
        self.num_envs = self.cfg.env.num_envs
        self.num_actions = self.cfg.env.num_actions
        self.num_dof = self.cfg.env.num_dof
        self.num_bodies = self.cfg.env.num_bodies
        self.num_feet = self.cfg.env.num_feet
        self.feet_indices = torch.arange(self.num_feet, device=self.device)
        self.termination_contact_indices = torch.tensor(
            self.cfg.env.termination_contact_indices, device=self.device, dtype=torch.long
        )
        self._init_buffers()
        self._init_actuator_network()
        self.obs_history_dim = self.cfg.env.num_observations * self.cfg.env.num_observation_history
        self.privileged_obs_dim = self.cfg.env.num_privileged_obs

    def summary(self) -> dict:
        """Return a summary for sanity checks in unit tests or CLI tools."""
        return {
            "num_envs": self.cfg.env.num_envs,
            "num_observations": self.cfg.env.num_observations,
            "obs_history_dim": self.obs_history_dim,
            "privileged_obs_dim": self.privileged_obs_dim,
            "control_decimation": self.cfg.control.decimation,
        }

    def _init_buffers(self) -> None:
        """Initialize torch buffers that mirror go1_gym/envs/base/legged_robot.py."""
        self.root_states = torch.zeros(self.num_envs, 13, device=self.device)
        self.dof_state = torch.zeros(self.num_envs, self.num_dof, 2, device=self.device)
        self.dof_pos = self.dof_state[..., 0]
        self.dof_vel = self.dof_state[..., 1]
        self.contact_forces = torch.zeros(self.num_envs, self.num_bodies, 3, device=self.device)
        self.rigid_body_state = torch.zeros(self.num_envs, self.num_bodies, 13, device=self.device)
        self.foot_positions = self.rigid_body_state[:, self.feet_indices, 0:3]
        self.foot_velocities = self.rigid_body_state[:, self.feet_indices, 7:10]

        self.base_pos = self.root_states[:, 0:3]
        self.base_quat = self.root_states[:, 3:7]
        self.base_lin_vel = self.root_states[:, 7:10]
        self.base_ang_vel = self.root_states[:, 10:13]
        self.gravity_vec = torch.tensor([0.0, 0.0, -1.0], device=self.device).repeat(self.num_envs, 1)
        self.projected_gravity = self.gravity_vec.clone()

        self.actions = torch.zeros(self.num_envs, self.num_actions, device=self.device)
        self.last_actions = torch.zeros_like(self.actions)
        self.last_last_actions = torch.zeros_like(self.actions)
        self.torques = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.torque_limits = torch.ones(self.num_dof, device=self.device)

        self.default_dof_pos = torch.zeros(1, self.num_dof, device=self.device)
        self.joint_pos_target = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, device=self.device)
        self.last_last_joint_pos_target = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        self.p_gains = torch.ones(self.num_dof, device=self.device)
        self.d_gains = torch.ones(self.num_dof, device=self.device)
        self.Kp_factors = torch.ones(self.num_envs, self.num_dof, device=self.device)
        self.Kd_factors = torch.ones(self.num_envs, self.num_dof, device=self.device)
        self.motor_strengths = torch.ones(self.num_envs, self.num_dof, device=self.device)
        self.motor_offsets = torch.zeros(self.num_envs, self.num_dof, device=self.device)

        self.commands = torch.zeros(self.num_envs, self.cfg.commands.num_commands, device=self.device)
        self.commands_scale = torch.ones(self.cfg.commands.num_commands, device=self.device)
        self.clock_inputs = torch.zeros(self.num_envs, 4, device=self.device)

        self.obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_observations, device=self.device)
        self.privileged_obs_buf = torch.zeros(self.num_envs, self.cfg.env.num_privileged_obs, device=self.device)
        self.obs_history = torch.zeros(
            self.num_envs,
            self.cfg.env.num_observations * self.cfg.env.num_observation_history,
            device=self.device,
        )

        self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.time_out_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self.episode_length_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.rew_buf = torch.zeros(self.num_envs, device=self.device)

        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()

        self.terrain_levels = torch.full(
            (self.num_envs,),
            self.cfg.terrain.min_init_terrain_level,
            device=self.device,
            dtype=torch.long,
        )
        self.extras: dict[str, dict] = {}

    def _init_actuator_network(self) -> None:
        self.actuator_network = None
        if self.cfg.control.control_type != "actuator_net":
            return
        actuator_path = Path(__file__).resolve().parents[2] / "resources" / "actuator_nets" / "unitree_go1.pt"
        actuator_network = torch.jit.load(str(actuator_path)).to(self.device)

        def eval_actuator_network(
            joint_pos: torch.Tensor,
            joint_pos_last: torch.Tensor,
            joint_pos_last_last: torch.Tensor,
            joint_vel: torch.Tensor,
            joint_vel_last: torch.Tensor,
            joint_vel_last_last: torch.Tensor,
        ) -> torch.Tensor:
            xs = torch.cat(
                (
                    joint_pos.unsqueeze(-1),
                    joint_pos_last.unsqueeze(-1),
                    joint_pos_last_last.unsqueeze(-1),
                    joint_vel.unsqueeze(-1),
                    joint_vel_last.unsqueeze(-1),
                    joint_vel_last_last.unsqueeze(-1),
                ),
                dim=-1,
            )
            torques = actuator_network(xs.view(self.num_envs * self.num_dof, 6))
            return torques.view(self.num_envs, self.num_dof)

        self.actuator_network = eval_actuator_network
        self.joint_pos_err_last_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.joint_pos_err_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.joint_vel_last_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)
        self.joint_vel_last = torch.zeros((self.num_envs, self.num_dof), device=self.device)

    def step(self, actions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        clip_actions = self.cfg.control.clip_actions
        self.actions = torch.clip(actions, -clip_actions, clip_actions).to(self.device)
        self.prev_base_pos = self.base_pos.clone()
        self.prev_base_quat = self.base_quat.clone()
        self.prev_base_lin_vel = self.base_lin_vel.clone()
        self.prev_foot_velocities = self.foot_velocities.clone()
        for _ in range(self.cfg.control.decimation):
            self.torques = self._compute_torques(self.actions)
        self.post_physics_step()
        return self.obs_buf, self.privileged_obs_buf, self.rew_buf, self.reset_buf, self.extras

    def post_physics_step(self) -> None:
        self.episode_length_buf += 1
        self.base_pos[:] = self.root_states[:, 0:3]
        self.base_quat[:] = self.root_states[:, 3:7]
        self.base_lin_vel[:] = self.root_states[:, 7:10]
        self.base_ang_vel[:] = self.root_states[:, 10:13]
        self.projected_gravity[:] = self.gravity_vec
        self.foot_velocities[:] = self.rigid_body_state[:, self.feet_indices, 7:10]
        self.foot_positions[:] = self.rigid_body_state[:, self.feet_indices, 0:3]

        self.check_termination()
        env_ids = self.reset_buf.nonzero(as_tuple=False).flatten()
        self.reset_idx(env_ids)
        self.compute_observations()

        self.last_last_actions[:] = self.last_actions
        self.last_actions[:] = self.actions
        self.last_last_joint_pos_target[:] = self.last_joint_pos_target
        self.last_joint_pos_target[:] = self.joint_pos_target

        self.obs_history = torch.cat((self.obs_history[:, self.cfg.env.num_observations :], self.obs_buf), dim=-1)

    def check_termination(self) -> None:
        if self.termination_contact_indices.numel() > 0:
            contact_norm = torch.norm(
                self.contact_forces[:, self.termination_contact_indices, :], dim=-1
            )
            self.reset_buf = torch.any(contact_norm > 1.0, dim=1)
        else:
            self.reset_buf = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        self.time_out_buf = self.episode_length_buf > self.cfg.env.max_episode_length
        self.reset_buf |= self.time_out_buf

        if self.cfg.env.use_terminal_body_height:
            self.body_height_buf = self.root_states[:, 2] < self.cfg.env.terminal_body_height
            self.reset_buf |= self.body_height_buf

    def reset_idx(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self._resample_commands(env_ids)
        self._reset_dofs(env_ids)
        self._reset_root_states(env_ids)

        self.last_actions[env_ids] = 0.0
        self.last_last_actions[env_ids] = 0.0
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        self.obs_history[env_ids] = 0.0

        self.extras.setdefault("train/episode", {})
        if self.cfg.terrain.curriculum:
            self._update_terrain_curriculum(env_ids)
            self.extras["train/episode"]["terrain_level"] = torch.mean(self.terrain_levels.float())

    def _update_terrain_curriculum(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        self.terrain_levels[env_ids] = torch.randint(
            self.cfg.terrain.min_init_terrain_level,
            self.cfg.terrain.max_init_terrain_level + 1,
            (env_ids.numel(),),
            device=self.device,
        )

    def _resample_commands(self, env_ids: torch.Tensor) -> None:
        ranges = torch.tensor(
            [
                self.cfg.commands.lin_vel_x,
                self.cfg.commands.lin_vel_y,
                self.cfg.commands.ang_vel_yaw,
                self.cfg.commands.body_height_cmd,
                self.cfg.commands.gait_frequency_cmd_range,
                self.cfg.commands.gait_phase_cmd_range,
                self.cfg.commands.gait_offset_cmd_range,
                self.cfg.commands.gait_bound_cmd_range,
                self.cfg.commands.gait_duration_cmd_range,
                self.cfg.commands.footswing_height_range,
                self.cfg.commands.body_pitch_range,
                self.cfg.commands.body_roll_range,
                self.cfg.commands.stance_width_range,
                self.cfg.commands.stance_length_range,
            ],
            device=self.device,
            dtype=torch.float,
        )
        if ranges.shape[0] < self.cfg.commands.num_commands:
            pad = self.cfg.commands.num_commands - ranges.shape[0]
            ranges = torch.cat((ranges, torch.zeros(pad, 2, device=self.device)), dim=0)
        low = ranges[: self.cfg.commands.num_commands, 0]
        high = ranges[: self.cfg.commands.num_commands, 1]
        self.commands[env_ids] = torch.rand(env_ids.numel(), self.cfg.commands.num_commands, device=self.device) * (
            high - low
        ) + low

    def _reset_dofs(self, env_ids: torch.Tensor) -> None:
        self.dof_pos[env_ids] = self.default_dof_pos.repeat(env_ids.numel(), 1)
        self.dof_vel[env_ids] = 0.0

    def _reset_root_states(self, env_ids: torch.Tensor) -> None:
        self.root_states[env_ids] = 0.0
        self.root_states[env_ids, 2] = 0.3

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        actions_scaled = actions[:, : self.num_dof] * self.cfg.control.action_scale
        if self.num_dof >= 10:
            actions_scaled[:, [0, 3, 6, 9]] *= self.cfg.control.hip_scale_reduction

        self.joint_pos_target = actions_scaled + self.default_dof_pos

        if self.cfg.control.control_type == "actuator_net":
            joint_pos_err = self.dof_pos - self.joint_pos_target + self.motor_offsets
            joint_vel = self.dof_vel
            torques = self.actuator_network(
                joint_pos_err,
                self.joint_pos_err_last,
                self.joint_pos_err_last_last,
                joint_vel,
                self.joint_vel_last,
                self.joint_vel_last_last,
            )
            self.joint_pos_err_last_last = self.joint_pos_err_last.clone()
            self.joint_pos_err_last = joint_pos_err.clone()
            self.joint_vel_last_last = self.joint_vel_last.clone()
            self.joint_vel_last = joint_vel.clone()
        elif self.cfg.control.control_type == "P":
            torques = self.p_gains * self.Kp_factors * (
                self.joint_pos_target - self.dof_pos + self.motor_offsets
            ) - self.d_gains * self.Kd_factors * self.dof_vel
        else:
            raise ValueError(f"Unknown controller type: {self.cfg.control.control_type}")

        torques = torques * self.motor_strengths
        return torch.clip(torques, -self.torque_limits, self.torque_limits)

    def compute_observations(self) -> None:
        components = [self.projected_gravity]
        if self.cfg.env.observe_gait_commands:
            components.append(self.commands * self.commands_scale)
        components.append(self.dof_pos)
        components.append(self.dof_vel)
        components.append(self.actions)

        if self.cfg.env.observe_two_prev_actions:
            components.append(self.last_actions)
        if self.cfg.env.observe_clock_inputs:
            components.append(self.clock_inputs)
        if self.cfg.env.observe_yaw:
            components.append(self.base_quat[:, 2:3])

        obs = torch.cat(components, dim=-1)
        self.obs_buf = self._align_observation_buffer(obs, self.cfg.env.num_observations)
        self.privileged_obs_buf = self._align_observation_buffer(
            self.privileged_obs_buf, self.cfg.env.num_privileged_obs
        )

    def _align_observation_buffer(self, obs: torch.Tensor, target_dim: int) -> torch.Tensor:
        if obs.shape[1] == target_dim:
            return obs
        if obs.shape[1] > target_dim:
            return obs[:, :target_dim]
        padding = torch.zeros(obs.shape[0], target_dim - obs.shape[1], device=obs.device)
        return torch.cat((obs, padding), dim=-1)
