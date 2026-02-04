"""IsaacLab task scaffolding for the Go1 legged robot."""

from dataclasses import asdict
from typing import Any, Dict, Optional

from .go1_config import Go1TaskCfg


class Go1LeggedRobotTask:
    """Skeleton IsaacLab task mirroring LeggedRobot behavior and buffers."""

    def __init__(self, cfg: Optional[Go1TaskCfg] = None, **kwargs: Any) -> None:
        self.cfg = cfg or Go1TaskCfg()
        self.kwargs = kwargs
        self._initialized = False
        self._buffers: Dict[str, Any] = {}
        self._extras: Dict[str, Any] = {}

    @classmethod
    def from_legged_robot_defaults(cls, **kwargs: Any) -> "Go1LeggedRobotTask":
        """Construct a task using defaults that mirror legged_robot_config.Cfg."""
        return cls(cfg=Go1TaskCfg(), **kwargs)

    def setup_scene(self) -> None:
        """Placeholder for IsaacLab scene creation (assets, terrain, sensors)."""
        self._initialized = True
        self._buffers = {
            "obs": None,
            "privileged_obs": None,
            "obs_history": None,
            "reward": None,
            "done": None,
        }

    def reset(self) -> Dict[str, Any]:
        """Reset the task and return the initial observations."""
        if not self._initialized:
            self.setup_scene()
        return {
            "obs": self._buffers["obs"],
            "privileged_obs": self._buffers["privileged_obs"],
            "obs_history": self._buffers["obs_history"],
        }

    def step(self, actions: Any) -> Dict[str, Any]:
        """Advance the simulation by one policy step."""
        if not self._initialized:
            self.setup_scene()
        return {
            "obs": self._buffers["obs"],
            "reward": self._buffers["reward"],
            "done": self._buffers["done"],
            "info": self._extras,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the task configuration for logging or debugging."""
        return asdict(self.cfg)
