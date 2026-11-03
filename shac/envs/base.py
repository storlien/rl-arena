"""
Abstract interface for differentiable, batched environments.

The SHAC algorithm treats the simulator as an abstract differentiable map

    s_{t+1} = F(s_t, a_t)                       (paper Eq. 1)

whose backward pass provides

    dL/ds_t = (dL/ds_{t+1}) * (dF/ds_t)
    dL/da_t = (dL/ds_{t+1}) * (dF/da_t)

In our pure-PyTorch environments this backward pass comes for free from
autograd: `step()` only uses differentiable tensor ops, so calling
`loss.backward()` through a sequence of steps computes exactly the adjoint
chain above.

Design notes
------------
* BATCHED: one environment instance simulates `num_envs` independent copies
  in parallel (one tensor row per copy). This mirrors the paper's parallel
  differentiable simulation (N = 64 trajectories sampled concurrently).

* GRADIENT DISCIPLINE: a rollout accumulates a potentially huge graph. SHAC
  deliberately *cuts* gradients between learning episodes (paper Section
  3.3), which is what `initialize_trajectory()` is for. Inside one short
  horizon (h = 32 steps) the graph is kept.

* MPC-FRIENDLY: the interface exposes `save_state()` / `load_state()` so a
  future model-predictive-control agent can fork simulated rollouts from the
  current state (candidate action sequences) without touching training.
  Gradient-based shooting through `step()` works out of the box because the
  dynamics are differentiable; sampling-based MPC (CEM/MPPI) can wrap
  `step()` in `torch.no_grad()` for speed.
"""

from typing import Dict, Optional
import torch


class DifferentiableEnv:
    """Base class for batched differentiable environments."""

    def __init__(self, num_envs: int, num_obs: int, num_actions: int,
                 episode_length: int, device: torch.device):
        self.num_envs = num_envs
        self.num_obs = num_obs
        self.num_actions = num_actions
        self.episode_length = episode_length   # task horizon H (steps)
        self.device = device

        # How many steps each environment copy has taken since its last
        # reset. Used to trigger horizon resets and to distinguish
        # "time limit reached" from (future tasks') early termination.
        self.progress_buf = torch.zeros(num_envs, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ API
    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset the given subset of environments (all if None) to fresh
        initial states drawn from the initial-state distribution.

        Returns the observation batch [num_envs, num_obs].
        Reset states are constants (leaf tensors): gradients never flow
        across a reset."""
        raise NotImplementedError

    def step(self, actions: torch.Tensor):
        """Advance one policy step (dt), applying `actions` [num_envs, A].

        Returns:
            obs    : [num_envs, num_obs] -- the observation AFTER the step.
                     For environments that reset on this step, this is the
                     observation of the *freshly reset* state (this matches
                     the official implementation).
            reward : [num_envs]          -- single-step reward R(s, a) of the
                     transition that just happened (pre-reset state).
            done   : [num_envs] bool    -- True if the task episode ended on
                     this step (the env has already been reset).
            info   : dict with 'obs_before_reset' -- the observation of the
                     pre-reset state for done envs (needed by SHAC to
                     bootstrap value at the time limit; see trainer).
        """
        raise NotImplementedError

    def initialize_trajectory(self) -> torch.Tensor:
        """Start a new gradient window from the *current* physical states,
        but DETACHED from all previous computation.

        Paper Section 3.3: "we note that the gradients are cut off between
        learning episodes to prevent unstable gradients during long-horizon
        backpropagation."  This is the method that performs the cut."""
        raise NotImplementedError

    def save_state(self) -> Dict[str, torch.Tensor]:
        """Snapshot of the full simulator state (for MPC rollouts / eval)."""
        raise NotImplementedError

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        """Restore a snapshot produced by `save_state`."""
        raise NotImplementedError

    # ------------------------------------------------------------ helpers
    @property
    def unwrapped(self) -> "DifferentiableEnv":
        return self
