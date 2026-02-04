from __future__ import annotations

from dataclasses import dataclass

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
