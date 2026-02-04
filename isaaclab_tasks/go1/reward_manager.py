from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict

import torch
from isaacgym.torch_utils import quat_conjugate, quat_from_angle_axis, quat_mul, quat_rotate_inverse

from go1_gym.utils.math_utils import quat_apply_yaw
from isaaclab_tasks.go1.config import Go1RewardScalesCfg


@dataclass
class Go1RewardParams:
    """Configuration parameters for reward shaping."""

    tracking_sigma: float
    tracking_sigma_yaw: float
    gait_force_sigma: float
    gait_vel_sigma: float
    base_height_target: float
    max_contact_force: float


@dataclass
class Go1RewardState:
    """State tensors required by the reward manager."""

    commands: torch.Tensor
    base_lin_vel: torch.Tensor
    base_ang_vel: torch.Tensor
    projected_gravity: torch.Tensor
    torques: torch.Tensor
    last_dof_vel: torch.Tensor
    dof_vel: torch.Tensor
    dt: float
    last_actions: torch.Tensor
    actions: torch.Tensor
    contact_forces: torch.Tensor
    penalised_contact_indices: torch.Tensor
    dof_pos: torch.Tensor
    dof_pos_limits: torch.Tensor
    base_pos: torch.Tensor
    desired_contact_states: torch.Tensor
    foot_velocities: torch.Tensor
    default_dof_pos: torch.Tensor
    joint_pos_target: torch.Tensor
    last_joint_pos_target: torch.Tensor
    last_last_joint_pos_target: torch.Tensor
    num_actuated_dof: int
    num_dof: int
    last_last_actions: torch.Tensor
    feet_indices: torch.Tensor
    last_contacts: torch.Tensor
    foot_positions: torch.Tensor
    prev_foot_velocities: torch.Tensor
    base_quat: torch.Tensor
    gravity_vec: torch.Tensor
    foot_indices: torch.Tensor

    def validate(self) -> None:
        missing = []
        for name in (
            "base_lin_vel",
            "base_ang_vel",
            "projected_gravity",
            "contact_forces",
            "foot_positions",
        ):
            if getattr(self, name, None) is None:
                missing.append(name)
        if missing:
            raise ValueError(f"Missing required reward tensors: {', '.join(missing)}")


RewardFn = Callable[[Go1RewardState, Go1RewardParams], torch.Tensor]


def reward_tracking_lin_vel(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    lin_vel_error = torch.sum(torch.square(state.commands[:, :2] - state.base_lin_vel[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / params.tracking_sigma)


def reward_tracking_ang_vel(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    ang_vel_error = torch.square(state.commands[:, 2] - state.base_ang_vel[:, 2])
    return torch.exp(-ang_vel_error / params.tracking_sigma_yaw)


def reward_lin_vel_z(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.square(state.base_lin_vel[:, 2])


def reward_ang_vel_xy(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(torch.square(state.base_ang_vel[:, :2]), dim=1)


def reward_orientation(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(torch.square(state.projected_gravity[:, :2]), dim=1)


def reward_torques(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(torch.square(state.torques), dim=1)


def reward_dof_acc(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(torch.square((state.last_dof_vel - state.dof_vel) / state.dt), dim=1)


def reward_action_rate(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(torch.square(state.last_actions - state.actions), dim=1)


def reward_collision(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(
        1.0 * (torch.norm(state.contact_forces[:, state.penalised_contact_indices, :], dim=-1) > 0.1), dim=1
    )


def reward_dof_pos_limits(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    out_of_limits = -(state.dof_pos - state.dof_pos_limits[:, 0]).clip(max=0.0)
    out_of_limits += (state.dof_pos - state.dof_pos_limits[:, 1]).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


def reward_jump(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    reference_heights = 0.0
    body_height = state.base_pos[:, 2] - reference_heights
    jump_height_target = state.commands[:, 3] + params.base_height_target
    return -torch.square(body_height - jump_height_target)


def reward_tracking_contacts_shaped_force(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    foot_forces = torch.norm(state.contact_forces[:, state.feet_indices, :], dim=-1)
    desired_contact = state.desired_contact_states
    reward = 0.0
    for i in range(4):
        reward += -(1 - desired_contact[:, i]) * (
            1 - torch.exp(-1.0 * foot_forces[:, i] ** 2 / params.gait_force_sigma)
        )
    return reward / 4.0


def reward_tracking_contacts_shaped_vel(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    foot_velocities = torch.norm(state.foot_velocities, dim=2).view(state.commands.shape[0], -1)
    desired_contact = state.desired_contact_states
    reward = 0.0
    for i in range(4):
        reward += -desired_contact[:, i] * (
            1 - torch.exp(-1.0 * foot_velocities[:, i] ** 2 / params.gait_vel_sigma)
        )
    return reward / 4.0


def reward_dof_pos(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(torch.square(state.dof_pos - state.default_dof_pos), dim=1)


def reward_dof_vel(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    return torch.sum(torch.square(state.dof_vel), dim=1)


def reward_action_smoothness_1(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    diff = torch.square(
        state.joint_pos_target[:, : state.num_actuated_dof]
        - state.last_joint_pos_target[:, : state.num_actuated_dof]
    )
    diff = diff * (state.last_actions[:, : state.num_dof] != 0)
    return torch.sum(diff, dim=1)


def reward_action_smoothness_2(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    diff = torch.square(
        state.joint_pos_target[:, : state.num_actuated_dof]
        - 2 * state.last_joint_pos_target[:, : state.num_actuated_dof]
        + state.last_last_joint_pos_target[:, : state.num_actuated_dof]
    )
    diff = diff * (state.last_actions[:, : state.num_dof] != 0)
    diff = diff * (state.last_last_actions[:, : state.num_dof] != 0)
    return torch.sum(diff, dim=1)


def reward_feet_slip(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    contact = state.contact_forces[:, state.feet_indices, 2] > 1.0
    contact_filt = torch.logical_or(contact, state.last_contacts)
    state.last_contacts = contact
    foot_velocities = torch.square(
        torch.norm(state.foot_velocities[:, :, 0:2], dim=2).view(state.commands.shape[0], -1)
    )
    return torch.sum(contact_filt * foot_velocities, dim=1)


def reward_feet_contact_vel(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    reference_heights = 0.0
    near_ground = state.foot_positions[:, :, 2] - reference_heights < 0.03
    foot_velocities = torch.square(
        torch.norm(state.foot_velocities[:, :, 0:3], dim=2).view(state.commands.shape[0], -1)
    )
    return torch.sum(near_ground * foot_velocities, dim=1)


def reward_feet_contact_forces(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    return torch.sum(
        (torch.norm(state.contact_forces[:, state.feet_indices, :], dim=-1) - params.max_contact_force).clip(min=0.0),
        dim=1,
    )


def reward_feet_clearance_cmd_linear(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    phases = 1 - torch.abs(1.0 - torch.clip((state.foot_indices * 2.0) - 1.0, 0.0, 1.0) * 2.0)
    foot_height = state.foot_positions[:, :, 2].view(state.commands.shape[0], -1)
    target_height = state.commands[:, 9].unsqueeze(1) * phases + 0.02
    rew_foot_clearance = torch.square(target_height - foot_height) * (1 - state.desired_contact_states)
    return torch.sum(rew_foot_clearance, dim=1)


def reward_feet_impact_vel(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    prev_foot_velocities = state.prev_foot_velocities[:, :, 2].view(state.commands.shape[0], -1)
    contact_states = torch.norm(state.contact_forces[:, state.feet_indices, :], dim=-1) > 1.0
    rew_foot_impact_vel = contact_states * torch.square(torch.clip(prev_foot_velocities, -100, 0))
    return torch.sum(rew_foot_impact_vel, dim=1)


def reward_orientation_control(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    roll_pitch_commands = state.commands[:, 10:12]
    quat_roll = quat_from_angle_axis(
        -roll_pitch_commands[:, 1],
        torch.tensor([1, 0, 0], device=state.commands.device, dtype=torch.float),
    )
    quat_pitch = quat_from_angle_axis(
        -roll_pitch_commands[:, 0],
        torch.tensor([0, 1, 0], device=state.commands.device, dtype=torch.float),
    )
    desired_base_quat = quat_mul(quat_roll, quat_pitch)
    desired_projected_gravity = quat_rotate_inverse(desired_base_quat, state.gravity_vec)
    return torch.sum(torch.square(state.projected_gravity[:, :2] - desired_projected_gravity[:, :2]), dim=1)


def reward_raibert_heuristic(state: Go1RewardState, params: Go1RewardParams) -> torch.Tensor:
    del params
    cur_footsteps_translated = state.foot_positions - state.base_pos.unsqueeze(1)
    footsteps_in_body_frame = torch.zeros(state.base_pos.shape[0], 4, 3, device=state.base_pos.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = quat_apply_yaw(
            quat_conjugate(state.base_quat), cur_footsteps_translated[:, i, :]
        )

    if state.commands.shape[1] >= 13:
        desired_stance_width = state.commands[:, 12:13]
        desired_ys_nom = torch.cat(
            [
                desired_stance_width / 2,
                -desired_stance_width / 2,
                desired_stance_width / 2,
                -desired_stance_width / 2,
            ],
            dim=1,
        )
    else:
        desired_stance_width = 0.3
        desired_ys_nom = torch.tensor(
            [desired_stance_width / 2, -desired_stance_width / 2, desired_stance_width / 2, -desired_stance_width / 2],
            device=state.base_pos.device,
        ).unsqueeze(0)

    if state.commands.shape[1] >= 14:
        desired_stance_length = state.commands[:, 13:14]
        desired_xs_nom = torch.cat(
            [
                desired_stance_length / 2,
                desired_stance_length / 2,
                -desired_stance_length / 2,
                -desired_stance_length / 2,
            ],
            dim=1,
        )
    else:
        desired_stance_length = 0.45
        desired_xs_nom = torch.tensor(
            [desired_stance_length / 2, desired_stance_length / 2, -desired_stance_length / 2, -desired_stance_length / 2],
            device=state.base_pos.device,
        ).unsqueeze(0)

    phases = torch.abs(1.0 - (state.foot_indices * 2.0)) * 1.0 - 0.5
    frequencies = state.commands[:, 4]
    x_vel_des = state.commands[:, 0:1]
    yaw_vel_des = state.commands[:, 2:3]
    y_vel_des = yaw_vel_des * desired_stance_length / 2
    desired_ys_offset = phases * y_vel_des * (0.5 / frequencies.unsqueeze(1))
    desired_ys_offset[:, 2:4] *= -1
    desired_xs_offset = phases * x_vel_des * (0.5 / frequencies.unsqueeze(1))

    desired_ys_nom = desired_ys_nom + desired_ys_offset
    desired_xs_nom = desired_xs_nom + desired_xs_offset
    desired_footsteps_body_frame = torch.cat((desired_xs_nom.unsqueeze(2), desired_ys_nom.unsqueeze(2)), dim=2)
    err_raibert_heuristic = torch.abs(desired_footsteps_body_frame - footsteps_in_body_frame[:, :, 0:2])
    return torch.sum(torch.square(err_raibert_heuristic), dim=(1, 2))


REWARD_TERMS: Dict[str, RewardFn] = {
    "tracking_lin_vel": reward_tracking_lin_vel,
    "tracking_ang_vel": reward_tracking_ang_vel,
    "lin_vel_z": reward_lin_vel_z,
    "ang_vel_xy": reward_ang_vel_xy,
    "orientation": reward_orientation,
    "torques": reward_torques,
    "dof_acc": reward_dof_acc,
    "action_rate": reward_action_rate,
    "collision": reward_collision,
    "dof_pos_limits": reward_dof_pos_limits,
    "jump": reward_jump,
    "tracking_contacts_shaped_force": reward_tracking_contacts_shaped_force,
    "tracking_contacts_shaped_vel": reward_tracking_contacts_shaped_vel,
    "dof_pos": reward_dof_pos,
    "dof_vel": reward_dof_vel,
    "action_smoothness_1": reward_action_smoothness_1,
    "action_smoothness_2": reward_action_smoothness_2,
    "feet_slip": reward_feet_slip,
    "feet_contact_vel": reward_feet_contact_vel,
    "feet_contact_forces": reward_feet_contact_forces,
    "feet_clearance_cmd_linear": reward_feet_clearance_cmd_linear,
    "feet_impact_vel": reward_feet_impact_vel,
    "orientation_control": reward_orientation_control,
    "raibert_heuristic": reward_raibert_heuristic,
}


@dataclass
class Go1RewardManager:
    params: Go1RewardParams
    scales: Go1RewardScalesCfg

    def compute(self, state: Go1RewardState) -> Dict[str, torch.Tensor]:
        state.validate()
        rewards: Dict[str, torch.Tensor] = {}
        scale_dict = vars(self.scales)
        for name, fn in REWARD_TERMS.items():
            scale = scale_dict.get(name, 0.0)
            if scale == 0.0:
                continue
            rewards[name] = fn(state, self.params) * scale
        return rewards
