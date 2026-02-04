from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import torch


class ObservationHistoryWrapper:
    """Wrap an IsaacLab environment to provide observation history tensors."""

    def __init__(
        self,
        env: Any,
        obs_key: str = "obs",
        privileged_obs_key: str = "privileged_obs",
    ) -> None:
        self.env = env
        self.obs_key = obs_key
        self.privileged_obs_key = privileged_obs_key

        self.num_envs = getattr(env, "num_envs", None)
        if self.num_envs is None:
            raise AttributeError("Environment must define num_envs for history tracking.")

        self.num_obs = self._get_num_obs(env)
        self.obs_history_length = self._get_obs_history_length(env)
        self.num_obs_history = self.num_obs * self.obs_history_length

        self.obs_history = torch.zeros(
            self.num_envs,
            self.num_obs_history,
            dtype=torch.float,
            device=self._get_device(),
            requires_grad=False,
        )

    def step(self, action: torch.Tensor) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor, Dict[str, Any]]:
        obs, reward, done, info = self.env.step(action)
        privileged_obs = self._extract_privileged_obs(obs, info)
        obs_tensor = self._extract_obs_tensor(obs)
        self._update_history(obs_tensor)
        info = self._ensure_info(info, privileged_obs)
        return self._build_obs_dict(obs, privileged_obs), reward, done, info

    def reset(self) -> Dict[str, torch.Tensor]:
        obs = self.env.reset()
        privileged_obs = self._extract_privileged_obs(obs, None)
        self.obs_history.zero_()
        return self._build_obs_dict(obs, privileged_obs)

    def reset_idx(self, env_ids: torch.Tensor) -> Any:
        if not hasattr(self.env, "reset_idx"):
            raise AttributeError("Wrapped environment does not define reset_idx.")
        result = self.env.reset_idx(env_ids)
        self.obs_history[env_ids] = 0
        return result

    def get_observations(self) -> Dict[str, torch.Tensor]:
        if not hasattr(self.env, "get_observations"):
            raise AttributeError("Wrapped environment does not define get_observations.")
        obs = self.env.get_observations()
        privileged_obs = self._extract_privileged_obs(obs, None)
        obs_tensor = self._extract_obs_tensor(obs)
        self._update_history(obs_tensor)
        return self._build_obs_dict(obs, privileged_obs)

    def _update_history(self, obs: torch.Tensor) -> None:
        self.obs_history = torch.cat((self.obs_history[:, self.num_obs :], obs), dim=-1)

    def _get_device(self) -> torch.device:
        device = getattr(self.env, "device", None)
        return device if device is not None else torch.device("cpu")

    def _extract_obs_tensor(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, dict):
            obs_tensor = obs[self.obs_key]
        else:
            obs_tensor = obs
        return torch.as_tensor(obs_tensor, device=self.obs_history.device)

    def _extract_privileged_obs(self, obs: Any, info: Optional[Dict[str, Any]]) -> Optional[torch.Tensor]:
        if isinstance(obs, dict) and self.privileged_obs_key in obs:
            return obs[self.privileged_obs_key]
        if info is not None and self.privileged_obs_key in info:
            return info[self.privileged_obs_key]
        if hasattr(self.env, "get_privileged_observations"):
            return self.env.get_privileged_observations()
        return None

    def _ensure_info(
        self, info: Optional[Dict[str, Any]], privileged_obs: Optional[torch.Tensor]
    ) -> Dict[str, Any]:
        info_dict: Dict[str, Any] = {} if info is None else dict(info)
        if privileged_obs is not None:
            info_dict[self.privileged_obs_key] = privileged_obs
        return info_dict

    def _build_obs_dict(self, obs: Any, privileged_obs: Optional[torch.Tensor]) -> Dict[str, torch.Tensor]:
        obs_dict: Dict[str, torch.Tensor] = {}
        if isinstance(obs, dict):
            obs_dict.update(obs)
            obs_tensor = obs[self.obs_key]
        else:
            obs_tensor = obs
        obs_dict[self.obs_key] = obs_tensor
        obs_dict[self.privileged_obs_key] = privileged_obs
        obs_dict["obs_history"] = self.obs_history
        return obs_dict

    @staticmethod
    def _get_num_obs(env: Any) -> int:
        for attr in ("num_obs", "num_observations"):
            if hasattr(env, attr):
                return int(getattr(env, attr))
        if hasattr(env, "cfg") and hasattr(env.cfg, "env"):
            return int(env.cfg.env.num_observations)
        raise AttributeError("Environment must define num_obs or num_observations.")

    @staticmethod
    def _get_obs_history_length(env: Any) -> int:
        if hasattr(env, "cfg") and hasattr(env.cfg, "env"):
            return int(env.cfg.env.num_observation_history)
        if hasattr(env, "num_observation_history"):
            return int(env.num_observation_history)
        raise AttributeError("Environment must define num_observation_history.")
