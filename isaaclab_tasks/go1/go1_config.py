"""IsaacLab config scaffolding mirroring go1_gym.envs.base.legged_robot_config.Cfg."""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


@dataclass
class EnvCfg:
    """Environment sizing and observation settings."""

    num_envs: int = 4096
    num_observations: int = 235
    num_scalar_observations: int = 42
    num_privileged_obs: int = 18
    num_actions: int = 12
    num_observation_history: int = 15
    env_spacing: float = 3.0
    send_timeouts: bool = True
    episode_length_s: float = 20.0
    observe_vel: bool = True
    observe_only_ang_vel: bool = False
    observe_only_lin_vel: bool = False
    observe_yaw: bool = False
    observe_contact_states: bool = False
    observe_command: bool = True
    observe_height_command: bool = False
    observe_gait_commands: bool = False
    observe_clock_inputs: bool = False
    observe_two_prev_actions: bool = False
    observe_imu: bool = False
    record_video: bool = True
    recording_width_px: int = 360
    recording_height_px: int = 240
    recording_mode: str = "COLOR"
    num_recording_envs: int = 1
    debug_viz: bool = False
    all_agents_share: bool = False

    priv_observe_friction: bool = True
    priv_observe_friction_indep: bool = True
    priv_observe_ground_friction: bool = False
    priv_observe_ground_friction_per_foot: bool = False
    priv_observe_restitution: bool = True
    priv_observe_base_mass: bool = True
    priv_observe_com_displacement: bool = True
    priv_observe_motor_strength: bool = False
    priv_observe_motor_offset: bool = False
    priv_observe_joint_friction: bool = True
    priv_observe_kp_factor: bool = True
    priv_observe_kd_factor: bool = True
    priv_observe_contact_forces: bool = False
    priv_observe_contact_states: bool = False
    priv_observe_body_velocity: bool = False
    priv_observe_foot_height: bool = False
    priv_observe_body_height: bool = False
    priv_observe_gravity: bool = False
    priv_observe_terrain_type: bool = False
    priv_observe_clock_inputs: bool = False
    priv_observe_doubletime_clock_inputs: bool = False
    priv_observe_halftime_clock_inputs: bool = False
    priv_observe_desired_contact_states: bool = False
    priv_observe_dummy_variable: bool = False


@dataclass
class TerrainCfg:
    """Terrain options (heightfield/trimesh) and curriculum settings."""

    mesh_type: str = "trimesh"
    horizontal_scale: float = 0.1
    vertical_scale: float = 0.005
    border_size: float = 0.0
    curriculum: bool = True
    static_friction: float = 1.0
    dynamic_friction: float = 1.0
    restitution: float = 0.0
    terrain_noise_magnitude: float = 0.1
    terrain_smoothness: float = 0.005
    measure_heights: bool = True
    measured_points_x: List[float] = field(
        default_factory=lambda: [
            -0.8,
            -0.7,
            -0.6,
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
        ]
    )
    measured_points_y: List[float] = field(
        default_factory=lambda: [
            -0.5,
            -0.4,
            -0.3,
            -0.2,
            -0.1,
            0.0,
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
        ]
    )
    selected: bool = False
    terrain_kwargs: Optional[Dict[str, float]] = None
    min_init_terrain_level: int = 0
    max_init_terrain_level: int = 5
    terrain_length: float = 8.0
    terrain_width: float = 8.0
    num_rows: int = 10
    num_cols: int = 20
    terrain_proportions: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.35, 0.25, 0.2])
    slope_treshold: float = 0.75
    difficulty_scale: float = 1.0
    x_init_range: float = 1.0
    y_init_range: float = 1.0
    yaw_init_range: float = 0.0
    x_init_offset: float = 0.0
    y_init_offset: float = 0.0
    teleport_robots: bool = True
    teleport_thresh: float = 2.0
    max_platform_height: float = 0.2
    center_robots: bool = False
    center_span: int = 5


@dataclass
class CommandsCfg:
    """Command sampling ranges and curricula configuration."""

    num_commands: int = 3
    resampling_time: float = 10.0
    subsample_gait: bool = False
    gait_interval_s: float = 10.0
    vel_interval_s: float = 10.0
    jump_interval_s: float = 20.0
    jump_duration_s: float = 0.1
    jump_height: float = 0.3
    heading_command: bool = True
    global_reference: bool = False
    observe_accel: bool = False
    command_curriculum: bool = False
    max_reverse_curriculum: float = 1.0
    max_forward_curriculum: float = 1.0
    yaw_command_curriculum: bool = False
    max_yaw_curriculum: float = 1.0
    exclusive_command_sampling: bool = False
    distributional_commands: bool = False
    curriculum_type: str = "RewardThresholdCurriculum"
    lipschitz_threshold: float = 0.9
    num_lin_vel_bins: int = 20
    lin_vel_step: float = 0.3
    num_ang_vel_bins: int = 20
    ang_vel_step: float = 0.3
    distribution_update_extension_distance: int = 1
    curriculum_seed: int = 100
    lin_vel_x: Tuple[float, float] = (-1.0, 1.0)
    lin_vel_y: Tuple[float, float] = (-1.0, 1.0)
    ang_vel_yaw: Tuple[float, float] = (-1.0, 1.0)
    body_height_cmd: Tuple[float, float] = (-0.05, 0.05)
    impulse_height_commands: bool = False
    limit_vel_x: Tuple[float, float] = (-10.0, 10.0)
    limit_vel_y: Tuple[float, float] = (-0.6, 0.6)
    limit_vel_yaw: Tuple[float, float] = (-10.0, 10.0)
    limit_body_height: Tuple[float, float] = (-0.05, 0.05)
    limit_gait_phase: Tuple[float, float] = (0.0, 0.01)
    limit_gait_offset: Tuple[float, float] = (0.0, 0.01)
    limit_gait_bound: Tuple[float, float] = (0.0, 0.01)
    limit_gait_frequency: Tuple[float, float] = (2.0, 2.01)
    limit_gait_duration: Tuple[float, float] = (0.49, 0.5)
    limit_footswing_height: Tuple[float, float] = (0.06, 0.061)
    limit_body_pitch: Tuple[float, float] = (0.0, 0.01)
    limit_body_roll: Tuple[float, float] = (0.0, 0.01)
    limit_aux_reward_coef: Tuple[float, float] = (0.0, 0.01)
    limit_compliance: Tuple[float, float] = (0.0, 0.01)
    limit_stance_width: Tuple[float, float] = (0.0, 0.01)
    limit_stance_length: Tuple[float, float] = (0.0, 0.01)
    num_bins_vel_x: int = 25
    num_bins_vel_y: int = 3
    num_bins_vel_yaw: int = 25
    num_bins_body_height: int = 1
    num_bins_gait_frequency: int = 11
    num_bins_gait_phase: int = 11
    num_bins_gait_offset: int = 2
    num_bins_gait_bound: int = 2
    num_bins_gait_duration: int = 3
    num_bins_footswing_height: int = 1
    num_bins_body_pitch: int = 1
    num_bins_body_roll: int = 1
    num_bins_aux_reward_coef: int = 1
    num_bins_compliance: int = 1
    num_bins_stance_width: int = 1
    num_bins_stance_length: int = 1
    heading: Tuple[float, float] = (-3.14, 3.14)
    gait_phase_cmd_range: Tuple[float, float] = (0.0, 0.01)
    gait_offset_cmd_range: Tuple[float, float] = (0.0, 0.01)
    gait_bound_cmd_range: Tuple[float, float] = (0.0, 0.01)
    gait_frequency_cmd_range: Tuple[float, float] = (2.0, 2.01)
    gait_duration_cmd_range: Tuple[float, float] = (0.49, 0.5)
    footswing_height_range: Tuple[float, float] = (0.06, 0.061)
    body_pitch_range: Tuple[float, float] = (0.0, 0.01)
    body_roll_range: Tuple[float, float] = (0.0, 0.01)
    aux_reward_coef_range: Tuple[float, float] = (0.0, 0.01)
    compliance_range: Tuple[float, float] = (0.0, 0.01)
    stance_width_range: Tuple[float, float] = (0.0, 0.01)
    stance_length_range: Tuple[float, float] = (0.0, 0.01)
    exclusive_phase_offset: bool = True
    binary_phases: bool = False
    pacing_offset: bool = False
    balance_gait_distribution: bool = True
    gaitwise_curricula: bool = True


@dataclass
class CurriculumThresholdsCfg:
    tracking_lin_vel: float = 0.8
    tracking_ang_vel: float = 0.5
    tracking_contacts_shaped_force: float = 0.8
    tracking_contacts_shaped_vel: float = 0.8


@dataclass
class InitStateCfg:
    pos: Tuple[float, float, float] = (0.0, 0.0, 1.0)
    rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
    lin_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    ang_vel: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    default_joint_angles: Dict[str, float] = field(
        default_factory=lambda: {"joint_a": 0.0, "joint_b": 0.0}
    )


@dataclass
class ControlCfg:
    """Low-level control and decimation settings."""

    control_type: str = "actuator_net"
    stiffness: Dict[str, float] = field(default_factory=lambda: {"joint_a": 10.0, "joint_b": 15.0})
    damping: Dict[str, float] = field(default_factory=lambda: {"joint_a": 1.0, "joint_b": 1.5})
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
    default_dof_drive_mode: int = 3
    self_collisions: int = 0
    replace_cylinder_with_capsule: bool = True
    flip_visual_attachments: bool = True
    density: float = 0.001
    angular_damping: float = 0.0
    linear_damping: float = 0.0
    max_angular_velocity: float = 1000.0
    max_linear_velocity: float = 1000.0
    armature: float = 0.0


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
    only_positive_rewards: bool = False
    only_positive_rewards_ji22_style: bool = False
    sigma_rew_neg: float = 0.0
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
    friction_range: Tuple[float, float] = (0.0, 1.0)
    ground_friction_range: Tuple[float, float] = (0.0, 1.0)


@dataclass
class DomainRandCfg:
    """Domain randomization toggles to mirror the original config."""

    push_robots: bool = False
    randomize_friction: bool = True
    friction_range: Tuple[float, float] = (0.5, 1.5)
    randomize_restitution: bool = True
    restitution_range: Tuple[float, float] = (0.0, 0.5)
    randomize_base_mass: bool = False
    added_mass_range: Tuple[float, float] = (-1.0, 3.0)
    randomize_gravity: bool = False
    gravity_range: Tuple[float, float] = (-1.0, 1.0)
    gravity_rand_interval_s: float = 8.0
    gravity_impulse_duration: float = 0.99
    randomize_com_displacement: bool = False
    com_displacement_range: Tuple[float, float] = (-0.15, 0.15)
    randomize_ground_friction: bool = False
    ground_friction_range: Tuple[float, float] = (0.0, 0.0)
    randomize_motor_strength: bool = False
    motor_strength_range: Tuple[float, float] = (0.9, 1.1)
    randomize_motor_offset: bool = False
    motor_offset_range: Tuple[float, float] = (-0.02, 0.02)
    randomize_kp_factor: bool = False
    randomize_kd_factor: bool = False
    rand_interval_s: float = 4.0


@dataclass
class Go1TaskCfg:
    """Top-level IsaacLab config stub for Go1 locomotion."""

    env: EnvCfg = field(default_factory=EnvCfg)
    terrain: TerrainCfg = field(default_factory=TerrainCfg)
    commands: CommandsCfg = field(default_factory=CommandsCfg)
    curriculum_thresholds: CurriculumThresholdsCfg = field(default_factory=CurriculumThresholdsCfg)
    init_state: InitStateCfg = field(default_factory=InitStateCfg)
    control: ControlCfg = field(default_factory=ControlCfg)
    asset: AssetCfg = field(default_factory=AssetCfg)
    sim: SimCfg = field(default_factory=SimCfg)
    viewer: ViewerCfg = field(default_factory=ViewerCfg)
    rewards: RewardsCfg = field(default_factory=RewardsCfg)
    normalization: NormalizationCfg = field(default_factory=NormalizationCfg)
    domain_rand: DomainRandCfg = field(default_factory=DomainRandCfg)
