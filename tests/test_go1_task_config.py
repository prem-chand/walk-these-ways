from isaaclab_tasks.go1.config import Go1EnvCfg, Go1TaskCfg
from isaaclab_tasks.go1.task import Go1LocomotionTask


def test_go1_task_summary_dimensions():
    cfg = Go1TaskCfg()
    task = Go1LocomotionTask(cfg=cfg)

    summary = task.summary()

    assert summary["num_envs"] == cfg.env.num_envs
    assert summary["num_observations"] == cfg.env.num_observations
    assert summary["obs_history_dim"] == cfg.env.num_observations * cfg.env.num_observation_history
    assert summary["privileged_obs_dim"] == cfg.env.num_privileged_obs


def test_go1_task_validation_rejects_invalid_envs():
    cfg = Go1TaskCfg(env=Go1EnvCfg(num_envs=0))
    try:
        Go1LocomotionTask(cfg=cfg)
        assert False, "Expected validation error for invalid num_envs."
    except ValueError as exc:
        assert "num_envs" in str(exc)
