"""IsaacLab config scaffolding mirroring go1_gym.envs.base.legged_robot_config.Cfg."""

from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class EnvCfg:
    """Environment sizing and observation settings."""

    num_envs: int = 4096
    num_observations: int = 235
    num_scalar_observations: int = 42
    num_privileged_obs: int = 18
    num_actions: int = 12
    num_observation_history: int = 15
    episode_length_s: float = 20.0
    observe_yaw: bool = False
    observe_gait_commands: bool = False
    observe_clock_inputs: bool = False
    observe_two_prev_actions: bool = False
    record_video: bool = True


@dataclass
class TerrainCfg:
    """Terrain options (heightfield/trimesh) and curriculum settings."""

    mesh_type: str = "trimesh"
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 0.0
    curriculum: bool = True
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.35, 0.25, 0.2])
    slope_treshold: float = 0.75
    teleport_robots: bool = True
    teleport_thresh: float = 2.0


@dataclass
class CommandsCfg:
    """Command sampling ranges and curricula configuration."""

    num_commands: int = 3
    resampling_time: float = 10.0
    command_curriculum: bool = False
    distributional_commands: bool = False
    lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
    ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)
    body_height_cmd: Tuple[float, float] = (-0.05, 0.05)


@dataclass
class ControlCfg:
    """Low-level control and decimation settings."""

    control_type: str = "actuator_net"
    action_scale: float = 0.5
    hip_scale_reduction: float = 1.0
    decimation: int = 4


@dataclass
class AssetCfg:
    """Robot asset metadata to align with the Go1 URDF/USD pipeline."""

    file: str = ""
    foot_name: str = "None"
    penalize_contacts_on: List[str] = field(default_factory=list)
    terminate_after_contacts_on: List[str] = field(default_factory=list)
    fix_base_link: bool = False
    self_collisions: int = 0
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True


@dataclass
class SimCfg:
    """Simulation timestep and physics settings."""

    dt: float = 0.005
    substeps: int = 1
    gravity: Tuple[float, float, float] = (0.0, 0.0, -9.81)


@dataclass
class ViewerCfg:
    """Viewer configuration for headless/GUI runs."""

    enable_viewer: bool = True
    pos: Tuple[float, float, float] = (20.0, 20.0, 5.0)
    lookat: Tuple[float, float, float] = (0.0, 0.0, 1.0)


@dataclass
class RewardsCfg:
    """Reward configuration aligning with CoRL reward terms."""

    reward_container_name: str = "CoRLRewards"
    tracking_sigma: float = 0.25
    tracking_sigma_yaw: float = 0.25
    base_height_target: float = 0.0
    max_contact_force: float = 100.0
    reward_scales: Dict[str, float] = field(
        default_factory=lambda: {
            "tracking_lin_vel": 1.0,
            "tracking_ang_vel": 0.5,
            "orientation": -0.1,
            "torques": -0.0001,
        }
    )


@dataclass
class NormalizationCfg:
    """Observation/action normalization parameters."""

    clip_observations: float = 5.0
    clip_actions: float = 10.0


@dataclass
class DomainRandCfg:
    """Domain randomization toggles to mirror the original config."""

    push_robots: bool = False
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.5, 1.5)
    randomize_restitution: bool = True
    restitution_range: Tuple[float, float] = (0.0, 0.5)


@dataclass
class Go1TaskCfg:
    """Top-level IsaacLab config stub for Go1 locomotion."""

    env: EnvCfg = field(default_factory=EnvCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    asset: AssetCfg = field(default_factory=AssetCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    viewer: ViewerCfg = field(default_factory=ViewerCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    normalization: NormalizationCfg = field(default_factory=NormalizationCfg)
    domain_rand: DomainRandCfg = field(default_factory=DomainRandCfg)
