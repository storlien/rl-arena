"""
Environment registry.

The trainer (and the scripts) are environment-agnostic: they hold an env by
NAME and ask this module for the matching class and config dataclass. Adding
a new differentiable environment is then a three-line change here plus the
env module itself.
"""

from typing import Optional

import torch

from shac.config import EnvConfig, PendulumEnvConfig
from shac.envs.base import DifferentiableEnv
from shac.envs.cartpole_swingup import CartPoleSwingUpEnv
from shac.envs.pendulum import PendulumSwingUpEnv

# name -> environment class. The name is what appears on the command line
# (--env), in checkpoints ("env_name") and in run metadata.
ENV_REGISTRY = {
    "cartpole": CartPoleSwingUpEnv,
    "pendulum": PendulumSwingUpEnv,
}

# name -> the matching config dataclass. Each env keeps its own documented
# defaults (mirroring the paper / official configs where they exist), so the
# dataclasses are deliberately NOT merged into one.
CONFIG_REGISTRY = {
    "cartpole": EnvConfig,
    "pendulum": PendulumEnvConfig,
}


def make_env_config(name: str, **overrides):
    """Build the config dataclass for env `name` with CLI overrides applied."""
    if name not in CONFIG_REGISTRY:
        raise ValueError(
            f"unknown env '{name}' (available: {sorted(CONFIG_REGISTRY)})")
    return CONFIG_REGISTRY[name](**overrides)


def make_env(name: str, cfg, device: torch.device,
             num_envs: Optional[int] = None) -> DifferentiableEnv:
    """Instantiate env `name` from its config `cfg` (num_envs may be
    overridden, e.g. for the evaluator)."""
    if name not in ENV_REGISTRY:
        raise ValueError(
            f"unknown env '{name}' (available: {sorted(ENV_REGISTRY)})")
    return ENV_REGISTRY[name](cfg, device, num_envs=num_envs)
