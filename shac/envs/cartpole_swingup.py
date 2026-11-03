"""
Differentiable CartPole Swing-Up environment (paper Appendix A.3.1).

------------------------------------------------------------------------------
THE TASK
------------------------------------------------------------------------------
A pole is attached (hinge) on top of a cart that slides on a horizontal rail.
The pole starts (near) HANGING DOWN; the policy controls the force applied to
the cart and must swing the pole UP and balance it there for as long as
possible, while keeping the cart near x = 0.

    observation (5): [x, x_dot, sin(theta), cos(theta), theta_dot]   (Table 6)
    action      (1): force applied to the cart, in [-1, 1] * action_strength
    reward      :  R = -theta^2 - 0.1*theta_dot^2 - 0.05*x^2 - 0.1*x_dot^2
                                                                (paper Eq. 13)
    horizon     :  H = 240 policy steps, NO early termination.

theta is measured FROM THE UPRIGHT position (theta=0 is balanced up,
theta=+/-pi is hanging down). This matches the official dflex/URDF setup,
which initializes the pole at theta = -pi (hanging).

------------------------------------------------------------------------------
THE DYNAMICS (derived with the Euler-Lagrange equations)
------------------------------------------------------------------------------
Generalized coordinates:  q = (x, theta),  velocities qd = (x_dot, theta_dot)

The pole is a rigid body:
    m      = pole mass
    l      = distance from the pivot to the pole's center of mass
    I_com  = pole's rotational inertia about its own COM
    I      = I_com + m * l^2          (inertia about the pivot, parallel axis)
    M      = cart mass
    g      = gravity
    F      = force applied to the cart by the action

With the pole's COM at (x + l*sin(theta), l*cos(theta)):

    Kinetic energy:  T = 0.5 (M+m) x_dot^2  +  m*l*x_dot*theta_dot*cos(theta)
                         + 0.5 * I * theta_dot^2
    Potential:       V = m*g*l*cos(theta)

The Euler-Lagrange equations  d/dt(dL/dqd) - dL/dq = tau  give:

    (M+m) x_ddot  +  m*l*cos(theta) * theta_ddot
                  -  m*l*sin(theta) * theta_dot^2          =  F
                                  (1)

    m*l*cos(theta) * x_ddot  +  I * theta_ddot
                              -  m*g*l*sin(theta)           =  0
                                  (2)

This is a linear system in (x_ddot, theta_ddot):

    [ (M+m)     m*l*cos ] [ x_ddot     ]   [ F + m*l*sin(theta)*theta_dot^2 ]
    [ m*l*cos       I    ] [ theta_ddot ] = [ m*g*l*sin(theta)              ]

with determinant  det = (M+m)*I - (m*l*cos(theta))^2  > 0  (always, by
positive-definiteness of the mass matrix). Solving analytically:

    x_ddot     = ( I*(F + m*l*s*td^2) - m*l*c * m*g*l*s ) / det
    theta_ddot = ( (M+m)*m*g*l*s - m*l*c*(F + m*l*s*td^2) ) / det

where s = sin(theta), c = cos(theta), td = theta_dot. Both accelerations are
smooth, closed-form functions of (q, qd, F) -- hence trivially
differentiable with autograd, which is all SHAC needs (paper Eq. 1).

Sanity checks of the formulas:
  * theta = 0 (upright), F = 0, theta_dot = 0  ->  accelerations = 0:
    an (unstable) equilibrium.                                 [correct]
  * theta = pi (hanging), small force F -> cart accelerates while the
    pole tips backward, initially pivoting about its COM.      [correct]

------------------------------------------------------------------------------
INTEGRATION: semi-implicit (symplectic) Euler
------------------------------------------------------------------------------
The paper (Section 3.1) integrates the dynamics with semi-implicit Euler,
i.e. velocities FIRST, then positions with the UPDATED velocities:

    qd <- qd + dt * qdd
    q  <- q  + dt * qd

Following the official environment, one *policy* step advances dt = 1/60 s
using `sim_substeps` = 4 substeps of dt/4 = 1/240 s each (with the action
held constant and accelerations recomputed every substep). The action is
constant across substeps, so gradients flow through all of them.

------------------------------------------------------------------------------
JOINT LIMITS (paper Eq. 4)
------------------------------------------------------------------------------
The cart slider has travel limits [-4, 4] m (from the official URDF). Beyond
the limit a continuous penalty force pushes the cart back (paper Eq. 4):

    f_limit = k_limit * (q_limit - q)     (plus a small damping term, as in
                                           the official dflex simulator)

The pole's hinge is "continuous" (no limit). In practice this force is almost
never engaged because the reward already penalizes |x|.
"""

from typing import Dict, Optional
import math

import torch

from shac.envs.base import DifferentiableEnv
from shac.config import EnvConfig


class CartPoleSwingUpEnv(DifferentiableEnv):
    """A batched, fully differentiable CartPole Swing-Up."""

    def __init__(self, cfg: EnvConfig, device: torch.device,
                 num_envs: Optional[int] = None):
        # Allow callers (e.g. the evaluator, or a future MPC agent) to use
        # a different number of parallel copies with the same parameters.
        cfg = EnvConfig(**{**cfg.__dict__, "num_envs": num_envs or cfg.num_envs})
        super().__init__(
            num_envs=cfg.num_envs,
            num_obs=5,
            num_actions=1,
            episode_length=cfg.episode_length,
            device=device,
        )
        self.cfg = cfg
        self.dtype = torch.float32 if cfg.dtype == "float32" else torch.float64

        # Derived rigid-body constants (see module docstring):
        self.M = cfg.cart_mass
        self.m = cfg.pole_mass
        self.l = cfg.pole_com_distance
        self.I = cfg.pole_inertia_com + cfg.pole_mass * cfg.pole_com_distance ** 2
        self.g = cfg.gravity
        # Frequently used combinations, precomputed for speed/readability:
        self.ml = self.m * self.l
        self.mgl = self.m * self.g * self.l

        self.dt_sub = cfg.dt / cfg.sim_substeps

        # Fixed part of the initial state (official env): the pole hangs
        # straight down (theta = -pi) and everything else is at rest.
        self._init_q = torch.tensor([0.0, -math.pi],
                                    dtype=self.dtype, device=device)
        self._init_qd = torch.zeros(2, dtype=self.dtype, device=device)

        # Physical state. These tensors carry the autograd graph during a
        # learning episode; after a reset the freshly written rows are
        # constants (no graph) which is exactly the semantics we want.
        self.q = torch.zeros(self.num_envs, 2, dtype=self.dtype, device=device)
        self.qd = torch.zeros(self.num_envs, 2, dtype=self.dtype, device=device)

        self.reset()  # draw initial states for all envs

    # ==================================================================
    # Physics
    # ==================================================================
    def _accelerations(self, q: torch.Tensor, qd: torch.Tensor,
                       force: torch.Tensor) -> torch.Tensor:
        """Solve the equations of motion for (x_ddot, theta_ddot).

        q, qd:  [N, 2] generalized positions/velocities
        force:  [N]    force applied to the cart [N]
        returns [N, 2] accelerations.
        Pure tensor math -> fully differentiable (paper Eq. 1 backward pass).
        """
        x, theta = q[:, 0], q[:, 1]
        xd, thd = qd[:, 0], qd[:, 1]

        s, c = torch.sin(theta), torch.cos(theta)

        # --- soft joint-limit force on the cart (paper Eq. 4 + damping) ---
        # Active only when x is outside [-cart_limit, +cart_limit]:
        #   if x < -limit: f = k*( -limit - x) > 0  (pushes right)
        #   if x > +limit: f = k*( +limit - x) < 0  (pushes left)
        # plus a damping term opposing the velocity while engaged.
        violation = torch.where(
            x < -self.cfg.cart_limit, -self.cfg.cart_limit - x,
            torch.where(x > self.cfg.cart_limit,
                       self.cfg.cart_limit - x,
                       torch.zeros_like(x)))
        f_limit = (self.cfg.limit_stiffness * violation
                   - self.cfg.limit_damping * xd * (violation != 0.0))

        # --- solve the 2x2 linear system (see module docstring) ---
        rhs1 = force + f_limit + self.ml * s * thd.square()   # Eq. (1) RHS
        rhs2 = self.mgl * s                                   # Eq. (2) RHS

        det = (self.M + self.m) * self.I - (self.ml * c).square()
        # det > 0 always (positive-definite mass matrix), so a plain division
        # is safe and keeps the expression smooth (no branching).
        x_ddot = (self.I * rhs1 - self.ml * c * rhs2) / det
        th_ddot = ((self.M + self.m) * rhs2 - self.ml * c * rhs1) / det

        return torch.stack([x_ddot, th_ddot], dim=-1)

    def _integrate(self, q, qd, action):
        """Advance dt with `sim_substeps` semi-implicit Euler substeps.
        The action is held constant over the substeps; gradients flow through
        every substep."""
        force = (self.cfg.action_strength * action).squeeze(-1)  # [N]
        for _ in range(self.cfg.sim_substeps):
            qdd = self._accelerations(q, qd, force)
            qd = qd + self.dt_sub * qdd          # velocity first ...
            q = q + self.dt_sub * qd            # ... then position (symplectic)
        return q, qd

    # ==================================================================
    # Observation / reward
    # ==================================================================
    def _observe(self, q, qd) -> torch.Tensor:
        """obs = [x, x_dot, sin(theta), cos(theta), theta_dot]  (Table 6)

        sin/cos make the angle observation continuous at the +/-pi wrap,
        which keeps the policy input smooth when the pole swings through
        the bottom position."""
        x, theta = q[:, 0:1], q[:, 1:2]
        xd, thd = qd[:, 0:1], qd[:, 1:2]
        return torch.cat([x, xd, torch.sin(theta), torch.cos(theta), thd], dim=-1)

    def _reward(self, q, qd) -> torch.Tensor:
        """R = -theta^2 - 0.1*theta_dot^2 - 0.05*x^2 - 0.1*x_dot^2  (Eq. 13)

        theta is first wrapped to [-pi, pi] (atan2(sin, cos)) so that a pole
        that spins around is not rewarded for large accumulated angles --
        it is the *orientation* that matters. The reward is a smooth
        differentiable function of the state (requirement of the paper:
        "reward formulations ... are defined as differentiable functions").
        Maximizing it balances the pole upright at x = 0; its maximum (~0)
        is attained at the goal."""
        x, theta = q[:, 0], q[:, 1]
        xd, thd = qd[:, 0], qd[:, 1]
        theta_n = torch.atan2(torch.sin(theta), torch.cos(theta))
        return (-theta_n.square() * 1.0
                - thd.square() * 0.1
                - x.square() * 0.05
                - xd.square() * 0.1)

    # ==================================================================
    # Environment API
    # ==================================================================
    def _initial_states(self, n: int):
        """Draw n initial states from the initial-state distribution.

        Official env: fixed pose (x=0, theta=-pi hanging, velocities 0) plus
          positions += pi   * (U[0,1] - 0.5)
          velocities += 0.5  * (U[0,1] - 0.5)
        """
        q0 = self._init_q.unsqueeze(0).expand(n, 2).clone()
        qd0 = self._init_qd.unsqueeze(0).expand(n, 2).clone()
        if self.cfg.stochastic_init:
            q0 = q0 + math.pi * (torch.rand(n, 2, dtype=self.dtype,
                                            device=self.device) - 0.5)
            qd0 = qd0 + 0.5 * (torch.rand(n, 2, dtype=self.dtype,
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
        # MPC optimizer probing raw actions) it enforces the effort limit.
        actions = torch.clamp(actions, -1.0, 1.0)

        # ---- forward dynamics (graph-carrying) ----
        q, qd = self._integrate(self.q, self.qd, actions)
        self.progress_buf += 1

        # ---- observation & reward of the transition that just happened ----
        obs_pre = self._observe(q, qd)
        reward = self._reward(q, qd)

        # ---- episode bookkeeping: reset at the task horizon H ----
        # (CartPole has no early termination; paper A.3.1)
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
