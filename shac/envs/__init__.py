"""
Environment registry.

The trainer (and the scripts) are environment-agnostic: they hold an env by
NAME and ask this module for the matching class and config dataclass. Adding
a new differentiable environment is a change HERE (one EnvSpec entry) plus
the env module itself -- nothing else in the codebase is env-specific.

The EnvSpec carries everything the rest of the codebase displays about an
env alongside its class (banner text, reward summary), so CLI banners and
plots follow the registry instead of per-env dicts scattered elsewhere.
"""

from dataclasses import dataclass
from typing import Optional, Type

import torch

from shac.config import EnvConfig, EnvConfigBase, PendulumEnvConfig
from shac.envs.base import DifferentiableEnv
from shac.envs.cartpole_swingup import CartPoleSwingUpEnv
from shac.envs.pendulum import PendulumSwingUpEnv


@dataclass(frozen=True)
class EnvSpec:
    """One registry entry: everything the codebase knows about an env.

    cls         the DifferentiableEnv subclass
    config_cls  the config dataclass parameterizing it (a subclass of
                EnvConfigBase; make_env type-checks cfg against it, so a
                mismatched config/env pairing fails with a clear message
                instead of a TypeError deep inside the env constructor)
    label       human-readable task name for the training banner and plot
                titles
    reward_desc one-line reward formula for the training banner
    """

    cls: Type[DifferentiableEnv]
    config_cls: Type[EnvConfigBase]
    label: str
    reward_desc: str


# The name is what appears on the command line (--env), in checkpoints
# ("env_name") and in run metadata.
ENV_REGISTRY = {
    "cartpole": EnvSpec(
        cls=CartPoleSwingUpEnv,
        config_cls=EnvConfig,
        label="CartPole Swing-Up (paper Appendix A.3.1)",
        reward_desc="reward = -theta^2 - 0.1*thd^2 - 0.05*x^2 - 0.1*xd^2",
    ),
    "pendulum": EnvSpec(
        cls=PendulumSwingUpEnv,
        config_cls=PendulumEnvConfig,
        label="Pendulum Swing-Up (Gym Pendulum-v1 spec; see PENDULUM.md "
              "-- not a SHAC paper task)",
        reward_desc="reward = -theta^2 - 0.1*thd^2 - 0.001*u^2 (u = torque)",
    ),
}


def get_env_spec(name: str) -> EnvSpec:
    """Look up an env's full registry entry (raises with the valid names)."""
    if name not in ENV_REGISTRY:
        raise ValueError(
            f"unknown env '{name}' (available: {sorted(ENV_REGISTRY)})")
    return ENV_REGISTRY[name]


def make_env_config(name: str, **overrides) -> EnvConfigBase:
    """Build the config dataclass for env `name` with CLI overrides applied."""
    return get_env_spec(name).config_cls(**overrides)


def make_env(name: str, cfg: EnvConfigBase, device: torch.device,
             num_envs: Optional[int] = None) -> DifferentiableEnv:
    """Instantiate env `name` from its config `cfg` (num_envs may be
    overridden, e.g. for the evaluator).

    Raises a clear error if `cfg` is not the config type registered for
    `name` (the classic mistake of pairing a PendulumEnvConfig with
    name='cartpole' would otherwise surface as an unrelated TypeError
    about unexpected keywords deep inside the env constructor).
    """
    spec = get_env_spec(name)
    if not isinstance(cfg, spec.config_cls):
        raise TypeError(
            f"env '{name}' expects a {spec.config_cls.__name__}, got "
            f"{type(cfg).__name__}")
    return spec.cls(cfg, device, num_envs=num_envs)
