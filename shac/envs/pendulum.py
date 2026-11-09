"""
Differentiable Pendulum Swing-Up environment.

------------------------------------------------------------------------------
THE TASK
------------------------------------------------------------------------------
A single rigid link hangs from a frictionless pivot. The policy applies a
TORQUE at the pivot and must swing the link UP and balance it inverted,
while keeping the torque (and hence the energy injected) small.

    observation (3): [cos(theta), sin(theta), theta_dot]
    action      (1): torque applied at the pivot,
                     in [-1, 1] * action_strength
    reward      :  R = -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2)
    horizon     :  H = 240 policy steps, NO early termination.

The task spec follows the classic Gym "Pendulum-v1" conventions (which this
repo does NOT otherwise use -- see PENDULUM.md): theta = 0 is UPRIGHT,
theta = +/-pi is HANGING, and the reward penalizes angle, angular velocity,
and squared torque. The SIMULATION conventions follow the repo's CartPole:
semi-implicit Euler, dt = 1/60 s per policy step with 4 substeps, H = 240,
no early termination, and the cartpole-style initial-state noise.

------------------------------------------------------------------------------
THE DYNAMICS
------------------------------------------------------------------------------
Generalized coordinate:  theta (measured FROM THE UPRIGHT position),
velocity:                theta_dot.

The link is a rigid body:
    m      = link mass
    l      = distance from the pivot to the link's center of mass
    I      = rotational inertia about the pivot: I = I_com + m * l^2
    g      = gravity
    u      = torque applied at the pivot by the action

With the link's COM at (l*sin(theta), l*cos(theta)) -- height l*cos(theta)
above the pivot -- the rotational equation of motion about the pivot is
simply

    I * theta_ddot  =  u  +  m*g*l*sin(theta)
                                  (1)

(The cart's 2x2 mass-matrix solve of the CartPole collapses to a single
equation: there is no second DoF to couple to. Note the PLUS sign: with
theta measured from upright, gravity's generalized torque m*g*l*sin(theta)
drives the link AWAY from upright (unstable) and restores it toward
hanging (stable). This matches the CartPole's hinge equation (2), whose
gravity term m*g*l*sin(theta) has the same sign, and Gym Pendulum-v1's
theta_ddot = 3g/(2l) sin(theta) + 3u/(m l^2).)

Sanity checks of the formula:
  * theta = 0 (upright), u = 0, theta_dot = 0  ->  theta_ddot = 0:
    an (unstable) equilibrium; displaced slightly, gravity pushes it
    further over.                                             [correct]
  * theta = pi (hanging), u = 0 -> theta_ddot = 0: a STABLE equilibrium;
    displaced slightly, the m*g*l*sin(theta) term restores toward pi.
                                                                [correct]
  * Unforced (u = 0): dE/dt = 0 for E = 0.5*I*theta_dot^2 + m*g*l*cos(theta)
    (energy conservation; checked in the tests).

There are NO joint limits (the pivot is continuous, like the CartPole's
hinge) and NO friction/damping (as in Gym Pendulum-v1), so (1) is the whole
story: theta_ddot is a smooth, closed-form function of (theta, theta_dot, u)
-- trivially differentiable with autograd, which is all SHAC needs.

------------------------------------------------------------------------------
INTEGRATION: semi-implicit (symplectic) Euler
------------------------------------------------------------------------------
Identical to the CartPole (paper Section 3.1): one *policy* step advances
dt = 1/60 s using `sim_substeps` = 4 substeps of dt/4 = 1/240 s each, with
the action held constant across substeps; gradients flow through all of
them:

    theta_dot <- theta_dot + dt_sub * theta_ddot
    theta     <- theta     + dt_sub * theta_dot
"""

from typing import Dict, Optional
import math

import torch

from shac.envs.base import DifferentiableEnv
from shac.config import PendulumEnvConfig


class PendulumSwingUpEnv(DifferentiableEnv):
    """A batched, fully differentiable Pendulum Swing-Up."""

    def __init__(self, cfg: PendulumEnvConfig, device: torch.device,
                 num_envs: Optional[int] = None):
        # Allow callers (e.g. the evaluator, or a future MPC agent) to use
        # a different number of parallel copies with the same parameters.
        cfg = PendulumEnvConfig(**{**cfg.__dict__,
                                   "num_envs": num_envs or cfg.num_envs})
        super().__init__(
            num_envs=cfg.num_envs,
            num_obs=3,
            num_actions=1,
            episode_length=cfg.episode_length,
            device=device,
        )
        self.cfg = cfg
        self.dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

        # Derived rigid-body constants (see module docstring):
        self.m = cfg.link_mass
        self.l = cfg.link_com_distance
        self.I = cfg.link_inertia_com + cfg.link_mass * cfg.link_com_distance ** 2
        self.g = cfg.gravity
        # Frequently used combinations, precomputed for speed/readability:
        self.mgl = self.m * self.g * self.l

        self.dt_sub = cfg.dt / cfg.sim_substeps

        # Fixed part of the initial state: the link hangs straight down
        # (theta = -pi) and is at rest (mirror of the CartPole env).
        self._init_q = torch.tensor([-math.pi], dtype=self.dtype,
                                    device=device)
        self._init_qd = torch.zeros(1, dtype=self.dtype, device=device)

        # Physical state. These tensors carry the autograd graph during a
        # learning episode; after a reset the freshly written rows are
        # constants (no graph) which is exactly the semantics we want.
        self.q = torch.zeros(self.num_envs, 1, dtype=self.dtype,
                             device=device)
        self.qd = torch.zeros(self.num_envs, 1, dtype=self.dtype,
                              device=device)

        self.reset()  # draw initial states for all envs

    # ==================================================================
    # Physics
    # ==================================================================
    def _accelerations(self, q: torch.Tensor, qd: torch.Tensor,
                       torque: torch.Tensor) -> torch.Tensor:
        """Equation of motion (1): theta_ddot = (u + m*g*l*sin(theta)) / I.

        (PLUS sign on the gravity term -- theta is measured from upright,
        so gravity drives the link away from upright and restores it
        toward hanging. See the module docstring for the derivation and
        the sign checks; the tests pin this against energy conservation
        and the equilibria.)

        q, qd:   [N, 1] generalized position/velocity
        torque:  [N]    torque applied at the pivot [N m]
        returns  [N, 1] angular acceleration.
        Pure tensor math -> fully differentiable (paper Eq. 1 backward pass).
        """
        theta = q[:, 0]
        return ((torque + self.mgl * torch.sin(theta)) / self.I).unsqueeze(-1)

    def _integrate(self, q, qd, action):
        """Advance dt with `sim_substeps` semi-implicit Euler substeps.
        The torque is held constant over the substeps; gradients flow
        through every substep."""
        torque = (self.cfg.action_strength * action).squeeze(-1)  # [N m]
        for _ in range(self.cfg.sim_substeps):
            qdd = self._accelerations(q, qd, torque)
            qd = qd + self.dt_sub * qdd          # velocity first ...
            q = q + self.dt_sub * qd            # ... then position (symplectic)
        return q, qd

    # ==================================================================
    # Observation / reward
    # ==================================================================
    def _observe(self, q, qd) -> torch.Tensor:
        """obs = [cos(theta), sin(theta), theta_dot]  (Gym Pendulum-v1)

        cos/sin make the angle observation continuous at the +/-pi wrap,
        which keeps the policy input smooth when the link swings through
        the hanging position (same trick as the CartPole's sin/cos obs)."""
        theta, thd = q[:, 0:1], qd[:, 0:1]
        return torch.cat([torch.cos(theta), torch.sin(theta), thd], dim=-1)

    def _reward(self, q, qd, torque) -> torch.Tensor:
        """R = -(theta^2 + 0.1*theta_dot^2 + 0.001*u^2)  (Gym Pendulum-v1)

        theta is first wrapped to [-pi, pi] (atan2(sin, cos)) so that a
        link that spins around is not rewarded for large accumulated
        angles -- it is the *orientation* that matters. u is the PHYSICAL
        torque (after action_strength scaling), so the penalty is on
        Newton-meters regardless of the action scaling. The reward is a
        smooth differentiable function of the state and action
        (paper: "reward formulations ... are defined as differentiable
        functions"). Its maximum (~0) is attained balanced upright with
        zero torque."""
        theta = q[:, 0]
        thd = qd[:, 0]
        theta_n = torch.atan2(torch.sin(theta), torch.cos(theta))
        return -(theta_n.square()
                 + thd.square() * 0.1
                 + torque.square() * 0.001)

    # ==================================================================
    # Environment API
    # ==================================================================
    def _initial_states(self, n: int):
        """Draw n initial states from the initial-state distribution.

        CartPole-style (this repo's convention, NOT gym's uniform): fixed
          pose (theta = -pi hanging, at rest) plus uniform noise:
          theta     += (pi/2) * (U[0,1] - 0.5)   -> theta in [-3pi/2, -pi/2]
          theta_dot += 0.25   * (U[0,1] - 0.5)   -> in [-0.125, 0.125]
        i.e. the same relative spread as the CartPole (positions noise
        +/-pi/2 around the fixed pose, velocity noise +/-0.25), so every
        episode is a genuine swing-up from (near) the hanging position."""
        q0 = self._init_q.unsqueeze(0).expand(n, 1).clone()
        qd0 = self._init_qd.unsqueeze(0).expand(n, 1).clone()
        if self.cfg.stochastic_init:
            q0 = q0 + (math.pi / 2) * (torch.rand(n, 1, dtype=self.dtype,
                                                  device=self.device) - 0.5)
            qd0 = qd0 + 0.25 * (torch.rand(n, 1, dtype=self.dtype,
                                           device=self.device) - 0.5)
        return q0, qd0

    def reset(self, env_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Reset the given envs (all if None) to fresh initial states.

        The written states are constants -> the autograd graph for those
        rows restarts from a leaf (gradients never cross a reset)."""
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        n = env_ids.numel()

        q0, qd0 = self._initial_states(n)
        # index_copy keeps the op out-of-place on the graph: untouched rows
        # keep their history, replaced rows become constants.
        self.q = self.q.index_copy(0, env_ids, q0.to(self.dtype))
        self.qd = self.qd.index_copy(0, env_ids, qd0.to(self.dtype))
        self.progress_buf[env_ids] = 0

        return self._observe(self.q, self.qd)

    def step(self, actions: torch.Tensor):
        """Advance one policy step. See base class for the return contract."""
        actions = actions.reshape(self.num_envs, self.num_actions)
        # Safety clip to the actuator range. After the trainer's tanh squash
        # this is a no-op with clean gradients; for other callers (e.g. an
        # MPC optimizer probing raw actions) it enforces the torque limit.
        actions = torch.clamp(actions, -1.0, 1.0)

        # ---- forward dynamics (graph-carrying) ----
        q, qd = self._integrate(self.q, self.qd, actions)
        self.progress_buf += 1

        # ---- observation & reward of the transition that just happened ----
        obs_pre = self._observe(q, qd)
        reward = self._reward(q, qd,
                              (self.cfg.action_strength * actions).squeeze(-1))

        # ---- episode bookkeeping: reset at the task horizon H ----
        # (Pendulum has no early termination, like CartPole)
        done = self.progress_buf > (self.episode_length - 1)

        if bool(done.any()):
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            # Re-draw initial states for the finished envs and splice them
            # in. torch.where is elementwise and differentiable: for rows
            # that reset, the (constant) initial state replaces the graph
            # node, cleanly cutting that trajectory.
            q0, qd0 = self._initial_states(self.num_envs)
            keep = (~done).unsqueeze(-1)
            q = torch.where(keep, q, q0)
            qd = torch.where(keep, qd, qd0)
            self.progress_buf[done_ids] = 0

        self.q, self.qd = q, qd
        obs = self._observe(self.q, self.qd)   # post-reset observations

        info = {
            # The pre-reset observation of envs that finished their episode
            # on this step. SHAC uses it to bootstrap the value at the time
            # limit (see trainer: "timeout" vs "early termination").
            "obs_before_reset": obs_pre,
            "done": done,
        }
        return obs, reward, done, info

    def initialize_trajectory(self) -> torch.Tensor:
        """Continue from the current states but DETACHED from past gradients
        (the cut between learning episodes, paper Section 3.3)."""
        self.q = self.q.detach()
        self.qd = self.qd.detach()
        return self._observe(self.q, self.qd)

    # ==================================================================
    # State snapshot (used by evaluation and future MPC rollouts)
    # ==================================================================
    def save_state(self) -> Dict[str, torch.Tensor]:
        return {
            "q": self.q.detach().clone(),
            "qd": self.qd.detach().clone(),
            "progress_buf": self.progress_buf.clone(),
        }

    def load_state(self, state: Dict[str, torch.Tensor]) -> None:
        self.q = state["q"].clone()
        self.qd = state["qd"].clone()
        self.progress_buf = state["progress_buf"].clone()
