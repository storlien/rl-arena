"""
Shared checkpoint-loading helpers.

Checkpoint files produced by SHACTrainer.save() carry everything needed to
rebuild an inference policy: the env name, the full env config, the actor
weights, the actor construction parameters (algo_cfg) and the observation
normalizer. Previously the rebuild logic was duplicated in every script
that reads checkpoints (render_policy.py, policy_metrics.py,
plot_std_ab.py) with drifting fidelity; this module is the single
implementation all of them (and SHACTrainer.load) use.
"""

import torch

from shac.config import EnvConfigBase
from shac.envs import DifferentiableEnv, make_env_config
from shac.models.networks import GaussianActor
from shac.utils.running_stats import RunningMeanStd


def read_checkpoint(path: str, device="cpu"):
    """Load a checkpoint file -> (env_name, env_cfg, ckpt).

    `env_name` comes from the checkpoint ("env_name"); checkpoints from
    before the env registry existed are cartpole-only and carry no field,
    so they default to "cartpole".
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    env_name = ckpt.get("env_name", "cartpole")
    env_cfg: EnvConfigBase = make_env_config(env_name, **ckpt["env_cfg"])
    return env_name, env_cfg, ckpt


def build_actor_and_rms(ckpt: dict, env: DifferentiableEnv, device="cpu"):
    """Rebuild the GaussianActor (+ observation normalizer) from a
    checkpoint, sized from `env` (obs/action dims are env properties).

    Returns (actor, rms); rms is None when the run ran without observation
    normalization. The actor is in eval() mode.
    """
    algo = ckpt["algo_cfg"]
    actor = GaussianActor(env.num_obs, env.num_actions,
                          tuple(algo["actor_units"]),
                          algo["actor_logstd_init"],
                          state_std=algo.get("actor_state_std",
                                             False)).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    rms = None
    if ckpt["obs_rms"] is not None:
        rms = RunningMeanStd((env.num_obs,), device=device)
        rms.load_state_dict(ckpt["obs_rms"])
    return actor, rms
