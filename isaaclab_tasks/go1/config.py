from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Go1TerrainCfg:
    mesh_type: str = "trimesh"
    curriculum: bool = True
    min_init_terrain_level: int = 0
    max_init_terrain_level: int = 5
    num_rows: int = 30
    num_cols: int = 30
    terrain_width: float = 5.0
    terrain_length: float = 5.0
    horizontal_scale: float = 0.10
    vertical_scale: float = 0.005
    border_size: float = 0.0
    center_robots: bool = True
    center_span: int = 4
    terrain_proportions: Tuple[float, ...] = (
        0.2,
        0.2,
        0.2,
        0.1,
        0.1,
        0.1,
        0.1,
        0.0,
        0.0,
        0.0,
    )


@dataclass(frozen=True)
class Go1CommandsCfg:
    num_commands: int = 15
    lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: Tuple[float, float] = (-0.6, 0.6)
    ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)
    body_height_cmd: Tuple[float, float] = (-0.25, 0.15)
    gait_frequency_cmd_range: Tuple[float, float] = (2.0, 4.0)
    gait_phase_cmd_range: Tuple[float, float] = (0.0, 1.0)
    gait_offset_cmd_range: Tuple[float, float] = (0.0, 1.0)
    gait_bound_cmd_range: Tuple[float, float] = (0.0, 1.0)
    gait_duration_cmd_range: Tuple[float, float] = (0.5, 0.5)
    footswing_height_range: Tuple[float, float] = (0.03, 0.35)
    body_pitch_range: Tuple[float, float] = (-0.4, 0.4)
    body_roll_range: Tuple[float, float] = (-0.0, 0.0)
    stance_width_range: Tuple[float, float] = (0.10, 0.45)
    stance_length_range: Tuple[float, float] = (0.35, 0.45)


@dataclass(frozen=True)
class Go1ControlCfg:
    decimation: int = 4
    control_type: str = "actuator_net"
    action_scale: float = 0.5
    hip_scale_reduction: float = 1.0
    clip_actions: float = 1.0


@dataclass(frozen=True)
class Go1EnvCfg:
    num_envs: int = 4000
    max_episode_length: int = 1000
    num_actions: int = 12
    num_dof: int = 12
    num_bodies: int = 17
    num_feet: int = 4
    num_observations: int = 70
    num_privileged_obs: int = 2
    num_observation_history: int = 30
    observe_yaw: bool = False
    observe_gait_commands: bool = True
    observe_clock_inputs: bool = True
    observe_two_prev_actions: bool = True
    termination_contact_indices: Tuple[int, ...] = ()
    use_terminal_body_height: bool = False
    terminal_body_height: float = 0.0


@dataclass(frozen=True)
class Go1RewardScalesCfg:
    termination: float = -0.0
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    lin_vel_z: float = -2.0
    ang_vel_xy: float = -0.05
    orientation: float = -0.0
    torques: float = -0.00001
    dof_vel: float = -0.0
    dof_acc: float = -2.5e-7
    base_height: float = -0.0
    feet_air_time: float = 1.0
    collision: float = -1.0
    feet_stumble: float = -0.0
    action_rate: float = -0.01
    stand_still: float = -0.0
    tracking_lin_vel_lat: float = 0.0
    tracking_lin_vel_long: float = 0.0
    tracking_contacts: float = 0.0
    tracking_contacts_shaped: float = 0.0
    tracking_contacts_shaped_force: float = 0.0
    tracking_contacts_shaped_vel: float = 0.0
    jump: float = 0.0
    energy: float = 0.0
    energy_expenditure: float = 0.0
    survival: float = 0.0
    dof_pos_limits: float = 0.0
    feet_contact_forces: float = 0.0
    feet_slip: float = 0.0
    feet_clearance_cmd_linear: float = 0.0
    dof_pos: float = 0.0
    action_smoothness_1: float = 0.0
    action_smoothness_2: float = 0.0
    base_motion: float = 0.0
    feet_impact_vel: float = 0.0
    raibert_heuristic: float = 0.0
    estimation_bonus: float = 0.0
    feet_clearance: float = 0.0
    feet_clearance_cmd: float = 0.0
    orientation_control: float = 0.0
    tracking_stance_width: float = 0.0
    tracking_stance_length: float = 0.0
    hop_symmetry: float = 0.0


@dataclass(frozen=True)
class Go1TaskCfg:
    env: Go1EnvCfg = field(default_factory=Go1EnvCfg)
    control: Go1ControlCfg = field(default_factory=Go1ControlCfg)
    commands: Go1CommandsCfg = field(default_factory=Go1CommandsCfg)
    terrain: Go1TerrainCfg = field(default_factory=Go1TerrainCfg)
    rewards: Go1RewardScalesCfg = field(default_factory=Go1RewardScalesCfg)

    def validate(self) -> None:
        if self.env.num_observations <= 0:
            raise ValueError("num_observations must be positive.")
        if self.env.num_envs <= 0:
            raise ValueError("num_envs must be positive.")
        if self.env.num_actions <= 0:
            raise ValueError("num_actions must be positive.")
        if self.env.num_dof <= 0:
            raise ValueError("num_dof must be positive.")
        if self.env.num_bodies <= 0:
            raise ValueError("num_bodies must be positive.")
        if self.control.decimation <= 0:
            raise ValueError("control.decimation must be positive.")
        if self.terrain.num_rows <= 0 or self.terrain.num_cols <= 0:
            raise ValueError("terrain grid must be positive.")
        if self.commands.num_commands <= 0:
            raise ValueError("num_commands must be positive.")
