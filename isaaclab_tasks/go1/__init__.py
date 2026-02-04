"""Go1 IsaacLab task scaffolding."""

from isaaclab_tasks.go1.config import Go1TaskCfg
from isaaclab_tasks.go1.observation_history import ObservationHistoryWrapper
from isaaclab_tasks.go1.task import Go1LocomotionTask

__all__ = ["Go1LocomotionTask", "Go1TaskCfg", "ObservationHistoryWrapper"]
