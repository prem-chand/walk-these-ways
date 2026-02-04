from __future__ import annotations

from dataclasses import dataclass, field
from typing import Tuple


@dataclass(frozen=True)
class Go1TerrainCfg:
    mesh_type: str = "trimesh"
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


@dataclass(frozen=True)
class Go1EnvCfg:
    num_envs: int = 4000
    max_episode_length: int = 1000
    num_observations: int = 70
    num_privileged_obs: int = 2
    num_observation_history: int = 30
    observe_yaw: bool = False
    observe_gait_commands: bool = True
    observe_clock_inputs: bool = True
    observe_two_prev_actions: bool = True


@dataclass(frozen=True)
class Go1RewardScalesCfg:
    tracking_lin_vel: float = 1.0
    tracking_ang_vel: float = 0.5
    feet_slip: float = -0.04
    action_smoothness_1: float = -0.1
    action_smoothness_2: float = -0.1
    dof_vel: float = -1e-4
    dof_pos: float = -0.0
    jump: float = 10.0
    base_height: float = 0.0
    tracking_contacts_shaped_force: float = 4.0
    tracking_contacts_shaped_vel: float = 4.0
    collision: float = -5.0


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
        if self.control.decimation <= 0:
            raise ValueError("control.decimation must be positive.")
        if self.terrain.num_rows <= 0 or self.terrain.num_cols <= 0:
            raise ValueError("terrain grid must be positive.")
        if self.commands.num_commands <= 0:
            raise ValueError("num_commands must be positive.")
