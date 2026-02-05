from __future__ import annotations

import argparse
from pathlib import Path

import glob
import numpy as np
import torch

from omni.isaac.lab.app import AppLauncher
from omni.isaac.lab.envs import ManagerBasedRLEnv
from omni.isaac.lab_tasks.utils import parse_env_cfg


def load_policy(logdir: Path):
    body = torch.jit.load(str(logdir / "checkpoints" / "body_latest.jit"))
    adaptation_module = torch.jit.load(str(logdir / "checkpoints" / "adaptation_module_latest.jit"))

    def policy(obs, info=None):
        if info is None:
            info = {}
        obs_history = extract_obs_history(obs)
        latent = adaptation_module.forward(obs_history.to("cpu"))
        action = body.forward(torch.cat((obs_history.to("cpu"), latent), dim=-1))
        info["latent"] = latent
        return action

    return policy


def extract_obs_history(obs):
    if isinstance(obs, dict):
        if "obs_history" in obs:
            return obs["obs_history"]
        if "policy" in obs and isinstance(obs["policy"], dict) and "obs_history" in obs["policy"]:
            return obs["policy"]["obs_history"]
        if "observations" in obs and isinstance(obs["observations"], dict) and "obs_history" in obs["observations"]:
            return obs["observations"]["obs_history"]
    return obs


def resolve_logdir(label: str | None, logdir: str | None) -> Path:
    if logdir:
        return Path(logdir).expanduser().resolve()
    if not label:
        raise ValueError("Either --label or --logdir must be provided.")
    dirs = glob.glob(str(Path("..") / "runs" / label / "*"))
    if not dirs:
        raise FileNotFoundError(f"No runs found for label: {label}")
    return Path(sorted(dirs)[0]).resolve()


def get_command_tensor(env) -> torch.Tensor:
    if hasattr(env, "commands"):
        return env.commands
    if hasattr(env, "command_manager"):
        manager = env.command_manager
        if hasattr(manager, "command"):
            return manager.command
        if hasattr(manager, "commands"):
            return manager.commands
    raise AttributeError("Unable to locate command tensor on the IsaacLab environment.")


def get_base_lin_vel(env) -> torch.Tensor:
    if hasattr(env, "base_lin_vel"):
        return env.base_lin_vel
    if hasattr(env, "robot"):
        robot = env.robot
        if hasattr(robot, "data") and hasattr(robot.data, "root_lin_vel_w"):
            return robot.data.root_lin_vel_w
    if hasattr(env, "scene") and hasattr(env.scene, "get"):
        robot = env.scene.get("robot", None)
        if robot is not None and hasattr(robot, "data") and hasattr(robot.data, "root_lin_vel_w"):
            return robot.data.root_lin_vel_w
    raise AttributeError("Unable to locate base linear velocity on the IsaacLab environment.")


def get_joint_positions(env) -> torch.Tensor:
    if hasattr(env, "dof_pos"):
        return env.dof_pos
    if hasattr(env, "robot"):
        robot = env.robot
        if hasattr(robot, "data") and hasattr(robot.data, "joint_pos"):
            return robot.data.joint_pos
    if hasattr(env, "scene") and hasattr(env.scene, "get"):
        robot = env.scene.get("robot", None)
        if robot is not None and hasattr(robot, "data") and hasattr(robot.data, "joint_pos"):
            return robot.data.joint_pos
    raise AttributeError("Unable to locate joint positions on the IsaacLab environment.")


def get_env_dt(env) -> float:
    if hasattr(env, "dt"):
        return float(env.dt)
    if hasattr(env, "physics_dt"):
        return float(env.physics_dt)
    if hasattr(env, "sim") and hasattr(env.sim, "dt"):
        return float(env.sim.dt)
    return 1.0


def build_command_profile(device: torch.device) -> dict[str, torch.Tensor | float]:
    gaits = {
        "pronking": [0.0, 0.0, 0.0],
        "trotting": [0.5, 0.0, 0.0],
        "bounding": [0.0, 0.5, 0.0],
        "pacing": [0.0, 0.0, 0.5],
    }
    return {
        "x_vel_cmd": 1.5,
        "y_vel_cmd": 0.0,
        "yaw_vel_cmd": 0.0,
        "body_height_cmd": 0.0,
        "step_frequency_cmd": 3.0,
        "gait": torch.tensor(gaits["trotting"], device=device),
        "footswing_height_cmd": 0.08,
        "pitch_cmd": 0.0,
        "roll_cmd": 0.0,
        "stance_width_cmd": 0.25,
    }


def apply_command_profile(command_tensor: torch.Tensor, profile: dict[str, torch.Tensor | float]) -> None:
    command_tensor[:, 0] = profile["x_vel_cmd"]
    command_tensor[:, 1] = profile["y_vel_cmd"]
    command_tensor[:, 2] = profile["yaw_vel_cmd"]
    command_tensor[:, 3] = profile["body_height_cmd"]
    command_tensor[:, 4] = profile["step_frequency_cmd"]
    command_tensor[:, 5:8] = profile["gait"]
    command_tensor[:, 8] = 0.5
    command_tensor[:, 9] = profile["footswing_height_cmd"]
    command_tensor[:, 10] = profile["pitch_cmd"]
    command_tensor[:, 11] = profile["roll_cmd"]
    command_tensor[:, 12] = profile["stance_width_cmd"]


def run_evaluation(args: argparse.Namespace) -> None:
    app_launcher = AppLauncher(headless=args.headless)
    simulation_app = app_launcher.app

    env_cfg = parse_env_cfg(
        args.task,
        device=args.device,
        num_envs=args.num_envs,
        use_fabric=not args.disable_fabric,
    )
    env = ManagerBasedRLEnv(cfg=env_cfg)

    logdir = resolve_logdir(args.label, args.logdir)
    policy = load_policy(logdir)

    command_tensor = get_command_tensor(env)
    profile = build_command_profile(command_tensor.device)

    measured_x_vels = np.zeros(args.num_steps)
    target_x_vels = np.ones(args.num_steps) * profile["x_vel_cmd"]
    joint_positions = np.zeros((args.num_steps, args.num_joints))

    obs = env.reset()

    for step in range(args.num_steps):
        with torch.no_grad():
            actions = policy(obs)
        apply_command_profile(command_tensor, profile)
        obs, _, _, _ = env.step(actions)

        base_lin_vel = get_base_lin_vel(env)
        dof_pos = get_joint_positions(env)
        measured_x_vels[step] = base_lin_vel[0, 0].item()
        joint_positions[step] = dof_pos[0, : args.num_joints].detach().cpu().numpy()

    dt = get_env_dt(env)
    time = np.linspace(0.0, args.num_steps * dt, args.num_steps)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        time=time,
        measured_x_vels=measured_x_vels,
        target_x_vels=target_x_vels,
        joint_positions=joint_positions,
    )

    env.close()
    simulation_app.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="IsaacLab evaluation for Walk These Ways.")
    parser.add_argument("--task", type=str, default="Go1Locomotion", help="IsaacLab task registry name.")
    parser.add_argument("--label", type=str, default="gait-conditioned-agility/pretrain-v0/train")
    parser.add_argument("--logdir", type=str, default=None, help="Direct path to a training run logdir.")
    parser.add_argument("--num-steps", type=int, default=250)
    parser.add_argument("--num-joints", type=int, default=12)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--output", type=str, default="runs/isaaclab_eval_metrics.npz")
    parser.add_argument("--headless", action="store_true", help="Run without GUI rendering.")
    parser.add_argument("--disable-fabric", action="store_true", help="Disable Fabric for IsaacLab.")
    return parser.parse_args()


if __name__ == "__main__":
    run_evaluation(parse_args())
