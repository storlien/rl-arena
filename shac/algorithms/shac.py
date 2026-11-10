"""
Short-Horizon Actor-Critic (SHAC) -- paper Algorithm 1.

------------------------------------------------------------------------------
THE ALGORITHM IN ONE PARAGRAPH
------------------------------------------------------------------------------
SHAC is an on-policy actor-critic method that turns a DIFFERENTIABLE
simulator into accurate policy gradients. Each learning episode:

  1. ROLLOUT: sample N parallel trajectories of h steps (h << H), continuing
     from where the previous episode's trajectories ended, keeping the
     autograd graph INSIDE the window and cutting it BETWEEN episodes.

  2. ACTOR UPDATE (paper Eq. 5):

         L_theta = -1/(N h) * sum_i [ sum_{t=t0}^{t0+h-1} gamma^{t-t0} R(s_t,a_t)
                                      + gamma^h * V'(s_{t0+h}) ]

     i.e. discounted short-horizon reward + the TARGET critic's value of the
     window's terminal state. Backpropagation through the differentiable
     simulator yields the exact analytic gradient dL/dtheta (paper Eq. 8-12;
     the terminal adjoint of Eq. 9, dL/ds_{t0+h} = -gamma^h/(Nh) * dV'/ds,
     arises automatically because the target critic is applied to the
     graph-connected terminal state). One Adam step updates the policy.

     ABLATION (--no-critic): the terminal bootstrap term gamma^h * V' is
     dropped, leaving ONLY the discounted short-horizon reward -- the pure
     pathwise-derivative part of Eq. 5 -- and the critic stages 3-4 are
     skipped entirely (discounted short-horizon BPTT).

  3. CRITIC UPDATE (paper Eq. 6-7): estimate a target value for every state
     visited in the window with td-(lambda) (exponentially-weighted mixtures
     of k-step returns, Eq. 7), then fit V by regression (16 iterations over
     4 minibatches in the official configuration).

  4. TARGET UPDATE: V' <- alpha * V' + (1 - alpha) * V   (Polyak averaging).

The short horizon avoids exploding/vanishing gradients of full BPTT and the
terminal critic smooths the long-horizon landscape -- the two key ideas of
the paper (Sections 3.2-3.3, Figure 2).

------------------------------------------------------------------------------
DETAILS THE PAPER LEAVES OPEN (resolved from the official implementation,
NVlabs/DiffRL/algorithms/shac.py -- all marked "official:" below)
------------------------------------------------------------------------------
* DISCOUNT RESETS: the discount power gamma^{t-t0} in Eq. 5 counts steps from
  the start of the CURRENT TASK EPISODE segment (t0 is the window start); it
  resets to gamma^0 whenever an environment finishes an episode (paper: "we
  ... conduct special handling such as resetting the discount ratio when
  task termination happens").  Official: a per-env discount factor vector
  reset to 1 at every done.

* SEGMENTED LOSS: each env contributes its accumulated discounted return +
  terminal value (i) at the moment it finishes an episode, and (ii) at the
  end of the window for the still-running remainder.  Terminal values are
  zero for EARLY termination (episode ended by failure) but bootstrap the
  TARGET critic on the pre-reset observation for a TIME LIMIT (the task
  continues, only the counter resets).  The environments in this repo
  (CartPole, Pendulum) only ever hit time limits.

* ACTION SQUASH: the sampled Gaussian action is squashed with tanh before
  being applied (bounding it to (-1, 1)); gradients flow through the tanh.

* OBSERVATION NORMALIZATION statistics are frozen for the duration of one
  learning episode (so the whole window shares one normalization), while
  being updated online for use by subsequent episodes.

------------------------------------------------------------------------------
AHAC EXTENSION (config-gated; NOT part of native SHAC)
------------------------------------------------------------------------------
Georgiev et al., "Adaptive Horizon Actor-Critic" (ICML 2024) = SHAC plus
three deltas, all OFF by default (repo convention: single-knob arms are
"SHAC + AHAC-inspired <component>"; only --ahac = all three together is
"AHAC"). Everything above this note is native SHAC; every AHAC-inspired
addition below is marked. The deltas:

  1. ADAPTIVE HORIZON (SHACConfig.horizon_mode="adaptive"): the window
     length H adapts each episode by a Lagrangian dual on contact
     stiffness (their Eq. 9-10), clipped to [steps_min, steps_max] = [8, 64]
     (official ahac.yaml). Implementation notes from the official code
     (imgeorgiev/DiffRL, src/shac/algorithms/ahac.py) that the paper's
     equations do NOT show:
       * the dual variable lambda NEVER enters the actor loss -- the actor
         objective is plain SHAC Eq. 5; the constraint acts exclusively
         through the H update (H += lr * lambda.sum());
       * lambda is UNCLAMPED; on a contact-free env (cfs = 0 < C every
         step) it therefore drifts negative without bound and H ratchets
         to steps_min and stays there -- the mechanism's domain boundary,
         analyzed in AHAC.md;
       * stiffness cfs = ||contact_forces / max(|accel|, 1)|| per env-step
         (DifferentiableEnv.stiffness; contact-free envs return zeros).
  2. DOUBLE CRITIC (train.py --double-critic): two independent online
     critics, min-pooled bootstrap, NO Polyak target network. This needs
     NO new trainer code -- it is exactly the existing SAPO-inspired
     combination critic_ensemble=2 + use_target_critic=False +
     actor_bootstrap_pool="min" (SAPO design choice IV and AHAC delta 2
     are the same mechanism).
  3. ITERATIVE CRITIC (SHACConfig.iterative_critic): train the critic to
     convergence EVERY episode -- up to iterative_critic_max passes with
     early stopping when the last five pass losses flatline (official:
     max 64, tol 2e-1) -- instead of native SHAC's fixed 16 passes.
"""

from typing import Dict, List, Optional, Tuple
import copy

import torch
import torch.nn.functional as F

from shac.config import EnvConfigBase, SHACConfig, RunConfig
from shac.envs import make_env
from shac.models.networks import GaussianActor, Critic
from shac.utils.checkpoints import read_checkpoint
from shac.utils.running_stats import RunningMeanStd


# ===========================================================================
# td-(lambda) value estimation (paper Eq. 7, generalized to episode resets)
# ===========================================================================
def td_lambda_values(rew_buf: torch.Tensor,
                     next_value_buf: torch.Tensor,
                     done_mask: torch.Tensor,
                     gamma: float,
                     lam: float) -> torch.Tensor:
    """Compute the estimated value of every state in a short-horizon window.

    Inputs (all shaped [h, N], h = window length, N = envs):
      rew_buf[i]        reward r_i of the transition s_i -> s_{i+1}
      next_value_buf[i] V(s_{i+1}) from the TARGET critic -- with terminal
                        overrides already applied by the caller:
                        0 for early termination, V(pre-reset state) for a
                        time limit, V(s_h) for the window's last transition.
      done_mask[i]      1.0 if env i finished its task episode at step i.
                        IMPORTANT: done_mask[h-1] must be all ones (the
                        window end acts as a bootstrap boundary for every
                        env, like a "done" whose terminal value is V(s_h)).

    Returns: [h, N] estimated values V-tilde(s_i).

    ---------------------------------------------------------------------------
    WHAT THIS COMPUTES (paper Eq. 7)
    ---------------------------------------------------------------------------
    For a state s_t whose task-episode segment still has n = e - t + 1 rewards
    available (e = index of the segment's last step), the k-step return is

        G^k_t = ( sum_{j<k} gamma^j * r_{t+j} ) + gamma^k * V(s_{t+k})

    and Eq. 7 mixes them geometrically:

        V-tilde(s_t) = (1 - lambda) * sum_{k=1}^{n-1} lambda^{k-1} * G^k_t
                       + lambda^{n-1} * G^n_t

    (weights sum to 1:  (1-lambda)*(1-lambda^{n-1})/(1-lambda) + lambda^{n-1}).

    ---------------------------------------------------------------------------
    THE RECURSION (matches the official implementation exactly)
    ---------------------------------------------------------------------------
    Processing steps in REVERSE with two running accumulators per env:

        lam_pow <- lam_pow * lambda * (1 - done_i) + done_i    # = lambda^{e-i}
        A_i     <- (1 - done_i) * ( lambda*gamma*A_{i+1}
                                    + gamma * V(s_{i+1})
                                    + (1 - lam_pow)/(1 - lambda) * r_i )
        B_i     <- gamma * ( V(s_{i+1}) * done_i + B_{i+1} * (1 - done_i) ) + r_i
        V-tilde(s_i) = (1 - lambda) * A_i + lam_pow * B_i

    One can show by induction (see tests/test_td_lambda.py, which checks
    this against the naive k-loop of Eq. 7) that
        B_i = G^{n}_t      (the full n-step return)
        (1-lambda)*A_i = (1-lambda) * sum_{k<n} lambda^{k-1} G^k_t
    so the result is exactly Eq. 7, with segments truncated at resets and
    bootstrapping the terminal value supplied in `next_value_buf`.
    """
    h, n = rew_buf.shape
    A = torch.zeros(n, dtype=rew_buf.dtype, device=rew_buf.device)
    B = torch.zeros(n, dtype=rew_buf.dtype, device=rew_buf.device)
    lam_pow = torch.ones(n, dtype=rew_buf.dtype, device=rew_buf.device)
    out = torch.zeros_like(rew_buf)

    for i in reversed(range(h)):
        dm = done_mask[i]                                  # [N], 1.0/0.0
        lam_pow = lam_pow * lam * (1.0 - dm) + dm          # lambda^{e-i}
        A = (1.0 - dm) * (lam * gamma * A
                          + gamma * next_value_buf[i]
                          + (1.0 - lam_pow) / (1.0 - lam) * rew_buf[i])
        B = gamma * (next_value_buf[i] * dm + B * (1.0 - dm)) + rew_buf[i]
        out[i] = (1.0 - lam) * A + lam_pow * B
    return out


# ===========================================================================
# AHAC-INSPIRED pure helpers (horizon_mode="adaptive" / --iterative-critic;
# NOT native SHAC). Kept module-level and side-effect-free so the dual
# mechanics are unit-testable without building a trainer (tests/test_ahac.py).
# Exact transcription of the official update loop:
# imgeorgiev/DiffRL src/shac/algorithms/ahac.py --
#   self.lambd -= lambd_lr * (self.C - self.cfs.mean(-1))
#   self.H     += lambd_lr * self.lambd.sum()
#   self.H      = torch.clip(self.H, self.steps_min, self.steps_max)
#   (next buffers) self.lambd = self.lambd[0].repeat(self.steps_num)
# ===========================================================================
def ahac_dual_update(lambd: torch.Tensor, horizon: float,
                     cfs_mean: torch.Tensor, contact_threshold: float,
                     lambd_lr: float, steps_min: int,
                     steps_max: int) -> Tuple[torch.Tensor, float]:
    """One AHAC dual step; returns the updated (lambda, H).

    lambd:    [h] per-STEP dual variables (one per rollout step; official
              cfs.mean(-1) averages over ENVS, leaving a per-step vector).
    cfs_mean: [h] per-step mean stiffness over envs for the CURRENT window.
    Signs: a step BELOW threshold (cfs < C, the contact-free regime) makes
    (C - cfs) > 0, so lambda DECREASES -- unclamped, it drifts negative
    without bound -- and since H += lr * sum(lambda), H shrinks toward
    steps_min. Above-threshold steps push lambda (and H) the other way.
    """
    lambd = lambd - lambd_lr * (contact_threshold - cfs_mean)
    horizon = float(horizon + lambd_lr * float(lambd.sum()))
    horizon = min(max(horizon, float(steps_min)), float(steps_max))
    # Official init_buffers: lambda is re-tiled (lambda[0] repeated) to the
    # NEW window length for the next episode.
    lambd = lambd[0].repeat(round(horizon))
    return lambd, horizon


def ahac_critic_converged(last_pass_losses: List[float],
                          tol: float) -> bool:
    """AHAC-INSPIRED iterative-critic early-stop criterion (official:
    critic_iterations=None, `abs(np.diff(last_losses).mean()) < 2e-1`
    over a 5-element window of per-PASS mean losses). Needs a full window
    of 5 pass losses before it can declare convergence."""
    if len(last_pass_losses) < 5:
        return False
    w = last_pass_losses[-5:]
    diffs = [b - a for a, b in zip(w, w[1:])]
    return abs(sum(diffs) / len(diffs)) < tol


# ===========================================================================
# The trainer
# ===========================================================================
class SHACTrainer:
    """SHAC for a differentiable environment (paper Algorithm 1)."""

    def __init__(self, env_cfg: EnvConfigBase, algo_cfg: SHACConfig,
                 run_cfg: RunConfig, device: torch.device,
                 env_name: str = "cartpole", logger=None):
        # NOTE on defaults: env_name keeps a "cartpole" default so old
        # call sites (and the paper-faithful usage) keep working, but
        # always pass it explicitly for the pendulum or any new env.
        self.env_cfg = env_cfg
        self.cfg = algo_cfg
        self.run_cfg = run_cfg
        self.device = device
        self.logger = logger
        self.env_name = env_name

        # ---- environment (the parallel differentiable simulator) ----
        self.env = make_env(env_name, env_cfg, device)
        self.num_envs = self.env.num_envs
        self.num_obs = self.env.num_obs
        self.num_actions = self.env.num_actions
        self.h = self.cfg.short_horizon          # h (window length)
        self.H = self.env.episode_length         # task horizon

        # ---- AHAC-INSPIRED adaptive-horizon state (NOT native SHAC) ----
        # horizon_mode="fixed" (default): this block is inert, self.h stays
        # short_horizon, and the buffers use the native shape.
        # horizon_mode="adaptive" (AHAC delta 1): the window length self.h
        # CHANGES every episode (AHAC's H, a float rounded per episode);
        # buffers are sized for the LARGEST window (steps_max) and the
        # critic stage slices them to the current length. h_max is the
        # buffer capacity in either mode.
        if self.cfg.horizon_mode == "adaptive":
            self.adaptive_horizon = True
            self.h_max = self.cfg.steps_max
            self.horizon_float = float(self.cfg.steps_min)   # AHAC's H
            self.lambd = torch.zeros(self.cfg.steps_min, device=device)
            self.h = self.cfg.steps_min    # current window = round(H)
        else:
            self.adaptive_horizon = False
            self.h_max = self.cfg.short_horizon
            self.horizon_float = float(self.cfg.short_horizon)
            self.lambd = None

        # ---- actor and critics ----
        self.use_critic = self.cfg.use_critic
        self.actor = GaussianActor(self.num_obs, self.num_actions,
                                    self.cfg.actor_units,
                                    self.cfg.actor_logstd_init,
                                    state_std=self.cfg.actor_state_std,
                                    activation=self.cfg.activation
                                    ).to(device)
        if self.use_critic:
            # critic_ensemble=1 (default) reproduces the paper exactly; >=2
            # is SAPO's design choice IV (see SHACConfig.critic_ensemble).
            # self.critic / self.target_critic / self.critic_opt stay as
            # aliases of the FIRST member so the logging, checkpoint and
            # test code keep a single canonical handle.
            self.critics = [Critic(self.num_obs, self.cfg.critic_units,
                                   activation=self.cfg.activation).to(device)
                            for _ in range(self.cfg.critic_ensemble)]
            self.critic = self.critics[0]
            if self.cfg.use_target_critic:
                self.target_critics = [copy.deepcopy(c) for c in self.critics]
                for t in self.target_critics:
                    # Target PARAMETERS are frozen (updated only by Polyak
                    # averaging); gradients still flow through their INPUT
                    # (d V'/d s, needed for the terminal adjoint, paper Eq. 9)
                    # but never update the weights.
                    for p in t.parameters():
                        p.requires_grad_(False)
            else:
                # SAPO choice IV: bootstrap from the ONLINE critics
                # (min-pooled over the ensemble), no target network.
                self.target_critics = []
            self.target_critic = (self.target_critics[0]
                                  if self.target_critics else None)
        else:
            # Ablation (--no-critic): no critic, no td-(lambda), no Polyak.
            # The actor loss below omits the gamma^h * V'(s_h) bootstrap term,
            # leaving only the pathwise-derivative window reward of Eq. 5.
            self.critics = []
            self.critic = None
            self.target_critics = []
            self.target_critic = None

        # ---- optimizers (paper: Adam with the paper's betas; SAPO choice
        #      V swaps in AdamW -- identical lr/betas, decoupled decay) ----
        opt_cls = (torch.optim.Adam if self.cfg.optimizer == "adam"
                   else torch.optim.AdamW)
        if self.cfg.optimizer not in ("adam", "adamw"):
            raise ValueError(
                f"unknown optimizer '{self.cfg.optimizer}' (adam | adamw)")
        self.actor_opt = opt_cls(self.actor.parameters(),
                                 lr=self.cfg.actor_learning_rate,
                                 betas=self.cfg.adam_betas)
        self.critic_opts = ([opt_cls(c.parameters(),
                                     lr=self.cfg.critic_learning_rate,
                                     betas=self.cfg.adam_betas)
                             for c in self.critics] if self.use_critic
                            else [])
        self.critic_opt = self.critic_opts[0] if self.critic_opts else None

        # ---- observation normalization (running mean/std) ----
        self.obs_rms = (RunningMeanStd((self.num_obs,), device=device)
                        if self.cfg.obs_rms else None)

        # ---- rollout buffers (refilled every learning episode) ----
        # Sized to the CAPACITY h_max (native: short_horizon; AHAC-inspired
        # adaptive mode: steps_max -- see the block above).
        shape = (self.h_max, self.num_envs)
        self.obs_buf = torch.zeros(self.h_max, self.num_envs, self.num_obs,
                                   device=device)          # normalized s_t
        self.rew_buf = torch.zeros(*shape, device=device)  # r_t
        self.next_value_buf = torch.zeros(*shape, device=device)  # V'(s_{t+1})
        self.done_mask = torch.zeros(*shape, device=device)       # 1.0 at dones
        # AHAC-INSPIRED: per-step contact stiffness cfs[i] = stiffness of the
        # transition taken at step i (zeros unless horizon_mode="adaptive"
        # and the env reports contact; consumed ONLY by the dual update).
        self.cfs_buf = torch.zeros(*shape, device=device)

        # ---- per-env task-episode bookkeeping (for logging and for the
        #      timeout-vs-early-termination distinction) ----
        self.episode_steps = torch.zeros(self.num_envs, dtype=torch.long,
                                         device=device)
        self.episode_return = torch.zeros(self.num_envs, device=device)

        # ---- maximum-entropy objective (SHACConfig.max_entropy) ----
        # SAC's temperature alpha, tuned by the dual step (SAPO Eq. 21);
        # H-bar normalizes the entropy that enters the losses (their
        # design choice II), while the dual sees the RAW entropy.
        self.max_entropy = self.cfg.max_entropy
        self.entropy_target = (self.cfg.entropy_target
                               if self.cfg.entropy_target is not None
                               else -0.5 * self.num_actions)
        self.entropy_alpha = float(self.cfg.entropy_init_alpha)
        self._ent_norm_den = 2.0 * abs(self.entropy_target) or 1.0
        # Per-window bookkeeping (filled by _run_window):
        #   ent_buf[i]  the NORMALIZED per-step entropy hat{h}_t
        #               (alpha * ent_buf enters the critic's soft targets)
        #   _window_ent_mean  mean RAW entropy over the window (logging)
        #   _window_ent_dual  mean (h_raw - H-bar) over the window (dual)
        #   _window_ent_contrib  the entropy part's exact contribution to
        #               the returned actor loss (verification/tests)
        if self.max_entropy:
            self.ent_buf = torch.zeros(*shape, device=device)
        self._window_ent_mean = 0.0
        self._window_ent_dual = 0.0
        self._window_ent_contrib = 0.0

        # ---- loop state (advanced by train(), restored by load/resume) ----
        self.total_env_steps = 0
        self._episode = 0                    # last COMPLETED learning episode
        self._best_eval = -float("inf")
        # Task-episode returns completed since the last log row. Filled by
        # _rollout_and_update_actor; train() clears it after every row.
        self._completed_returns: List[float] = []

    # ======================================================================
    # Learning-rate schedule (official: linear decay toward `lr_final`)
    # ======================================================================
    def _lrs(self, episode: int) -> Tuple[float, float]:
        frac = min(1.0, episode / self.cfg.num_episodes)
        decay = lambda x0: (self.cfg.lr_final - x0) * frac + x0
        return (decay(self.cfg.actor_learning_rate),
                decay(self.cfg.critic_learning_rate))

    def _set_lrs(self, episode: int) -> Tuple[float, float]:
        a, c = self._lrs(episode)
        for g in self.actor_opt.param_groups:
            g["lr"] = a
        if self.critic_opt is not None:
            for g in self.critic_opt.param_groups:
                g["lr"] = c
        return a, c

    # ======================================================================
    # STAGE 1+2: rollout one short-horizon window and update the ACTOR
    # (paper Eq. 5, gradient via backprop through the differentiable sim)
    # ======================================================================
    def _pool_bootstrap(self, obs: torch.Tensor, reduction: str = "min") -> torch.Tensor:
        """Ensemble-pooled bootstrap value V(s) on normalized observations.

        `reduction`: "min" (SAC double-Q pessimism) or "avg" (SAPO's paper
        §4.2 IV uses the AVERAGE of the two critics in the ACTOR's window
        return, keeping min only for the critic's bootstrap targets --
        pessimism where values are regressed, unbiased gradient where the
        actor learns). Single-critic runs ignore the reduction.

        Which networks: the (frozen) Polyak TARGET critics by default;
        SAPO-inspired IV (use_target_critic=False): the ONLINE critics.
        """
        nets = self.target_critics if self.target_critics else self.critics
        vals = [c(obs).squeeze(-1) for c in nets]
        if len(vals) == 1:
            return vals[0]
        stacked = torch.stack(vals)
        if reduction == "min":
            return stacked.min(dim=0).values
        if reduction == "avg":
            return stacked.mean(dim=0)
        raise ValueError(f"unknown bootstrap pooling '{reduction}' (min | avg)")

    def _run_window(self) -> torch.Tensor:
        """Rollout ONE short-horizon window and return the Eq. 5 actor loss
        (normalized by N*h), with its autograd graph attached.

        Shared by the actor update (which backprops and steps) and by the
        gradient-estimator-variance measurement (which samples the
        estimator WITHOUT stepping the optimizer -- see
        _measure_grad_variance). Performs ALL per-window bookkeeping --
        critic buffers, running observation statistics, per-env episode
        step/return counters -- so both callers drive the same environment
        and normalizer streams; only the env-step accounting and the Adam
        step live in the training caller.
        """
        h, N = self.h, self.num_envs
        gamma = self.cfg.gamma

        # Freeze the observation normalizer for this window (see module
        # docstring): every network input below uses `rms` (the snapshot),
        # while `self.obs_rms` keeps ingesting fresh statistics.
        rms = self.obs_rms.snapshot() if self.obs_rms is not None else None

        def norm(obs):
            return rms.normalize(obs) if rms is not None else obs

        # --- per-env accumulators for the segmented discounted return ---
        # rew_acc[e]   = sum over the current task-episode segment of
        #                gamma^{t-a} * r_t   (a = segment start)
        # gamma_pow[e] = gamma^{steps since segment start}; reset to 1 when
        #                the env finishes an episode (official handling)
        # ent_acc[e]   = same segmentation for the ENTROPY part of the
        #                max-entropy return (SAPO Eq. 18): the entropy term
        #                is a per-step scalar exactly like the reward, so
        #                it accumulates with the identical gamma/done logic
        #                and both are drained into the loss together.
        rew_acc = torch.zeros(N, device=self.device)
        ent_acc = torch.zeros(N, device=self.device)
        gamma_pow = torch.ones(N, device=self.device)
        actor_loss = torch.zeros((), device=self.device)
        ent_raw_sum = 0.0        # detached sum of per-step mean raw h
        self._window_ent_contrib = 0.0   # this window's entropy part (tests)

        # Fresh policy_std accumulation window (state-std mode): the
        # diagnostics' sigma-mean below then covers exactly this window's
        # states, never the gradient-probe's or previous episodes'.
        self.actor.reset_std()

        # Cut gradients from previous learning episodes; continue physically.
        obs_raw = self.env.initialize_trajectory()
        if self.obs_rms is not None:
            self.obs_rms.update(obs_raw)
        obs = norm(obs_raw)

        for i in range(h):
            # ---- collect the state for critic training (constant) ----
            if self.use_critic:
                with torch.no_grad():
                    self.obs_buf[i] = obs.clone()

            # ---- action: reparameterized Gaussian sample, tanh-squashed ----
            if self.max_entropy:
                # SAPO Eq. 18: the window return additionally carries
                # alpha * hat{h}_t per step. hat{h} is the H-bar-normalized
                # entropy (their design choice II -- approximately [0, 1],
                # robust across reward scales); the alpha dual below sees
                # the RAW h. sample_with_entropy consumes the RNG exactly
                # like forward(), so the rollout stream is unchanged.
                u_sample, ent_dim = self.actor.sample_with_entropy(obs)
                action = torch.tanh(u_sample)
                ent_raw = ent_dim.sum(dim=-1)                     # [N]
                ent_hat = ((ent_raw + abs(self.entropy_target))
                           / self._ent_norm_den)                  # [N]
                with torch.no_grad():
                    ent_raw_sum += float(ent_raw.mean())
            else:
                action = torch.tanh(self.actor(obs))

            # ---- AHAC-INSPIRED: record the transition's contact stiffness
            #      (no_grad inside the env API; it feeds the horizon dual,
            #      never a gradient). Native SHAC never calls this -- the
            #      base-class default is exact zeros.
            if self.adaptive_horizon:
                self.cfs_buf[i] = self.env.stiffness(action)

            # ---- differentiable simulator step ----
            obs_raw, rew, done, info = self.env.step(action)
            if self.obs_rms is not None:
                self.obs_rms.update(obs_raw)
            obs = norm(obs_raw)

            # ---- task-episode bookkeeping ----
            self.episode_steps += 1
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)

            # ---- terminal value V'(s_{t+1}) for every env ----
            # Default: target critic on the (post-reset) next observation.
            # Ablation (--no-critic): nv is identically zero, so every
            # bootstrap term below vanishes and Eq. 5 degenerates to the
            # pure pathwise-derivative window reward.
            if self.use_critic:
                # Split pooling (SAPO §4.2 IV, faithful): the critic's
                # td-(lambda) targets always bootstrap from the MIN-pooled
                # ensemble (pessimism); the ACTOR's window loss uses
                # cfg.actor_bootstrap_pool ("min" default = our SHAC+IV-
                # style arm; "avg" = SAPO's actor-return pooling). With one
                # critic the two are identical.
                nv_t = self._pool_bootstrap(obs, "min")
                nv_a = (nv_t if self.cfg.actor_bootstrap_pool == "min"
                        else self._pool_bootstrap(
                            obs, self.cfg.actor_bootstrap_pool))
                if bool(done.any()):
                    # For envs that just finished an episode, replace the
                    # value:
                    #   * early termination -> 0   (the episode is a dead end)
                    #   * time limit        -> V'(pre-reset state): the task
                    #     would continue, so we bootstrap the state it died
                    #     on.
                    pre_reset = norm(info["obs_before_reset"])
                    nv_limit_t = self._pool_bootstrap(pre_reset, "min")
                    nv_limit_a = (nv_limit_t
                                  if self.cfg.actor_bootstrap_pool == "min"
                                  else self._pool_bootstrap(
                                      pre_reset,
                                      self.cfg.actor_bootstrap_pool))
                    at_limit = self.episode_steps >= self.H
                    nv_term_t = torch.where(at_limit, nv_limit_t,
                                            torch.zeros_like(nv_limit_t))
                    nv_term_a = torch.where(at_limit, nv_limit_a,
                                            torch.zeros_like(nv_limit_a))
                    nv_t = torch.where(done, nv_term_t, nv_t)
                    nv_a = torch.where(done, nv_term_a, nv_a)
                nv = nv_a
            else:
                nv = torch.zeros(N, device=self.device)

            # ---- accumulate the actor loss (paper Eq. 5, segmented; with
            #      --max-entropy the segment return additionally carries
            #      the entropy part, SAPO Eq. 18-19) ----
            # Every step extends the running discounted return of each env's
            # current segment ...
            rew_acc = rew_acc + gamma_pow * rew
            if self.max_entropy:
                ent_acc = ent_acc + gamma_pow * self.entropy_alpha * ent_hat
            seg_acc = rew_acc + ent_acc if self.max_entropy else rew_acc
            if i < h - 1:
                # ... and envs that finish an episode contribute their full
                # segment return + terminal value NOW:
                if bool(done.any()):
                    actor_loss = actor_loss - (
                        seg_acc[done_ids]
                        + gamma * gamma_pow[done_ids] * nv[done_ids]).sum()
                    if self.max_entropy:
                        with torch.no_grad():
                            self._window_ent_contrib += float(
                                ent_acc[done_ids].sum())
            else:
                # at the window end, ALL envs contribute their remaining
                # segment:  sum_t gamma^{t-a} r_t + gamma^{h-a} V'(s_h)
                # (with a=0 this is exactly paper Eq. 5).
                actor_loss = actor_loss - (seg_acc
                                           + gamma * gamma_pow * nv).sum()
                if self.max_entropy:
                    with torch.no_grad():
                        self._window_ent_contrib += float(ent_acc.sum())

            # Discount power advances; finished envs start fresh segments.
            gamma_pow = gamma_pow * gamma
            gamma_pow = torch.where(done, torch.ones_like(gamma_pow), gamma_pow)
            rew_acc = torch.where(done, torch.zeros_like(rew_acc), rew_acc)
            if self.max_entropy:
                ent_acc = torch.where(done, torch.zeros_like(ent_acc),
                                      ent_acc)

            # ---- store constants for the critic stage (nothing to store
            #      in the no-critic ablation -- stage 3 does not exist) ----
            if self.use_critic:
                with torch.no_grad():
                    self.rew_buf[i] = rew
                    if self.max_entropy:
                        # The critic fits SOFT values: its td-(lambda)
                        # targets use r + alpha * hat{h} (SAPO Eq. 20) --
                        # hat{h} here, alpha applied at target time (the
                        # post-dual alpha, per their pseudocode).
                        self.ent_buf[i] = ent_hat.detach()
                    self.next_value_buf[i] = nv_t.detach()
                    # The window's last step is a bootstrap boundary for
                    # every env (its "terminal value" is V'(s_h));
                    # official: done_mask is forced to 1 on the final step.
                    self.done_mask[i] = (done.float() if i < h - 1
                                         else torch.ones_like(done.float()))

            # ---- per-episode return logging (undiscounted, raw reward) ----
            with torch.no_grad():
                self.episode_return += rew
                if bool(done.any()):
                    self._completed_returns.extend(
                        self.episode_return[done_ids].tolist())
                    self.episode_return[done_ids] = 0.0
                    self.episode_steps[done_ids] = 0

        # Per-window raw-entropy stats: logging (policy_entropy) and the
        # alpha dual (mean(h - H-bar) over steps and envs).
        if self.max_entropy:
            self._window_ent_mean = ent_raw_sum / h
            self._window_ent_dual = self._window_ent_mean - self.entropy_target

        # Eq. 5 normalization by N*h (paper Algorithm 1).
        return actor_loss / (N * h)

    def _rollout_and_update_actor(self,
                                  update_actor: bool = True) -> Dict[str, float]:
        """ONE learning episode: window rollout + ONE Adam step (Alg. 1).

        update_actor=False is the TD3-style critic warmup
        (SHACConfig.critic_warmup): the window still rolls out (the critic
        needs on-policy data, and the obs_rms must track the same stream
        it would see in normal training) but the actor gets no gradient
        step and the alpha dual is skipped.
        """
        N, h = self.num_envs, self.h
        actor_loss = self._run_window()

        if update_actor:
            self.actor_opt.zero_grad()
            actor_loss.backward()

            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.actor.parameters(), self.cfg.grad_norm_clip)
            if not torch.isfinite(grad_norm):
                raise RuntimeError(
                    f"Non-finite actor gradient norm ({grad_norm}). The "
                    "simulation probably diverged -- check action bounds/dt.")
            self.actor_opt.step()
        else:
            grad_norm = None

        # ---- the alpha dual (SAC Eq. 21 as instantiated by SAPO) ----
        # Gradient-ascent on the dual objective J(alpha) = alpha*(H-bar - h)
        # pushes alpha UP when the window entropy h is BELOW the target
        # (explore more) and DOWN when it is above. The actor loss above
        # used the PRE-dual alpha; the critic's soft targets below see the
        # POST-dual value (SAPO's pseudocode order). Clamped at 0: the
        # entropy bonus must never become a penalty.
        alpha_used = self.entropy_alpha
        if self.max_entropy and update_actor:
            self.entropy_alpha = max(
                0.0, self.entropy_alpha - self.cfg.entropy_lr
                * self._window_ent_dual)

        self.total_env_steps += N * h
        return {
            "actor_loss": float(actor_loss.item()),
            # None keeps the CSV schema identical during warmup episodes.
            "actor_grad_norm": (float(grad_norm.item())
                                if grad_norm is not None else None),
            "policy_std": self.actor.std(),
            # alpha that entered THIS episode's actor loss (pre-dual);
            # None keeps the CSV schema identical when the flag is off.
            "entropy_alpha": (alpha_used if self.max_entropy else None),
            "policy_entropy": (self._window_ent_mean if self.max_entropy
                               else None),
        }

    # ======================================================================
    # DIAGNOSTIC: sample the actor-gradient ESTIMATOR at the current policy
    # (the variance quantity behind the paper's short-horizon motivation,
    # Sec. 3.2: BPTT gradient variance grows with horizon, h bounds it)
    # ======================================================================
    def _measure_grad_variance(self, k: int) -> Dict[str, float]:
        """Sample the actor-gradient estimator k times at the CURRENT
        theta and summarize its magnitude and variance.

        WHY EXTRA SAMPLES: one training episode yields exactly ONE draw
        g(theta, xi) of the gradient; consecutive training rows sample
        DIFFERENT thetas (Adam moved in between), so their spread mixes
        estimator noise with the drift of the true gradient as the policy
        learns -- the estimator's variance is not identifiable from the
        training log alone. Holding the policy at theta for k extra
        windows (NO optimizer step, `.grad` untouched --
        torch.autograd.grad) while the window randomness xi (exploration
        noise, env resets) is resampled gives i.i.d. samples of the
        estimator AT this point in training.

        Returns:
          actor_grad_mean_norm  ||E[g]||: mean gradient over the k
            samples -- the best available estimate of the TRUE gradient
            norm at theta (training itself only ever sees single draws).
          actor_grad_var  E[||g||^2] - ||E[g]||^2: trace of the empirical
            gradient covariance -- the scalar "gradient variance" of one
            window's estimate.

        The k windows advance the environment stream and observation
        statistics like any window, but are NOT added to total_env_steps:
        they contribute no learning, and keeping them out preserves the
        comparability of the env_steps axis across runs with and without
        this diagnostic (their wall-clock cost is still visible in
        wall_time_s).
        """
        params = list(self.actor.parameters())
        samples = []
        for _ in range(k):
            loss = self._run_window()
            grads = torch.autograd.grad(loss, params)
            # One flat float64 vector per window: the covariance trace is
            # the difference of two similar-size terms, so the reduction
            # is done in double precision to avoid float32 cancellation.
            g = torch.cat([t.reshape(-1) for t in grads]).to(torch.float64)
            if not bool(torch.isfinite(g).all()):
                raise RuntimeError(
                    "Non-finite actor gradient while sampling the "
                    "estimator variance. The simulation probably diverged "
                    "-- check action bounds/dt.")
            samples.append(g)
        G = torch.stack(samples)                       # [k, P]
        mean = G.mean(dim=0)                           # [P]
        mean_norm = float(torch.linalg.vector_norm(mean).item())
        var = float((G * G).sum(dim=1).mean().item()) - mean_norm ** 2
        return {"actor_grad_mean_norm": mean_norm,
                "actor_grad_var": max(var, 0.0)}

    # ======================================================================
    # STAGE 3: fit the critic to td-(lambda) targets (paper Eq. 6-7)
    # ======================================================================
    def _update_critic(self) -> Tuple[float, float]:
        """Stage 3. Returns (mean MSE loss, mean PRE-CLIP gradient norm)
        over the 16*4 = 64 inner gradient steps -- the pre-clip norm is
        what clip_grad_norm_ returns, matching actor_grad_norm."""
        # ---- target values V-tilde(s) for every state in the window ----
        # AHAC-INSPIRED (horizon_mode="adaptive"): only the first h_cur rows
        # of the capacity-sized buffers belong to THIS window (the rest is
        # stale from longer past windows); native fixed-h runs have
        # h_cur == h_max and every slice below is a no-op.
        h_cur = self.h
        with torch.no_grad():
            # SOFT values under --max-entropy (SAPO Eq. 20): the td-(lambda)
            # recursion runs on r + alpha * hat{h} instead of r, with the
            # POST-dual alpha (the dual step in _rollout_and_update_actor
            # already ran). The critic then estimates the entropy-
            # augmented return, matching what the actor's window loss
            # optimizes.
            if self.max_entropy:
                soft_rew = self.rew_buf[:h_cur] \
                    + self.entropy_alpha * self.ent_buf[:h_cur]
            else:
                soft_rew = self.rew_buf[:h_cur]
            targets = td_lambda_values(soft_rew,
                                       self.next_value_buf[:h_cur],
                                       self.done_mask[:h_cur],
                                       self.cfg.gamma,
                                       self.cfg.td_lambda)     # [h_cur, N]

        # Flatten time-major: states [h*N, obs], targets [h*N].
        obs_flat = self.obs_buf[:h_cur].reshape(-1, self.num_obs)
        tgt_flat = targets.reshape(-1)

        # Official minibatching: `critic_num_batch` CONTIGUOUS slices of the
        # time-major buffer (batch 0 = first h/num_batch steps of all envs,
        # ...), iterated `critic_iterations` times.
        batch_size = max(1, obs_flat.shape[0] // self.cfg.critic_num_batch)
        num_batches = (obs_flat.shape[0] + batch_size - 1) // batch_size

        # Outer passes over the episode's data. NATIVE SHAC: a fixed count
        # (critic_iterations, paper Table 4: 16). AHAC-INSPIRED
        # (--iterative-critic): train to CONVERGENCE every episode -- up to
        # iterative_critic_max passes, early-stopping when the last five
        # pass losses flatline (official: max 64, tol 2e-1). The per-pass
        # loss for the criterion is the mean over batches (and ensemble
        # members) of this pass.
        if self.cfg.iterative_critic:
            max_passes = self.cfg.iterative_critic_max
            last_pass_losses: List[float] = []
        else:
            max_passes = self.cfg.critic_iterations
            last_pass_losses = None

        total_loss, total_gnorm, steps = 0.0, 0.0, 0
        for _pass in range(max_passes):
            pass_loss_sum, pass_steps = 0.0, 0
            for b in range(num_batches):
                idx = slice(b * batch_size, min((b + 1) * batch_size,
                                                 obs_flat.shape[0]))
                # With an ensemble (SAPO choice IV) every critic regresses
                # onto the SAME targets (built from the min-pooled
                # bootstrap); the ensemble only diversifies the BOOTSTRAP
                # value, not the regression problem. Losses are averaged
                # for logging.
                for critic, opt in zip(self.critics, self.critic_opts):
                    pred = critic(obs_flat[idx]).squeeze(-1)
                    loss = F.mse_loss(pred, tgt_flat[idx])

                    opt.zero_grad()
                    loss.backward()
                    # Official "ugly fix": sanitize non-finite grads if the
                    # simulation ever produced NaN observations. This is a
                    # DELIBERATE asymmetry with the actor path, which RAISES
                    # on non-finite gradients: a critic MSE spike (e.g. at
                    # the lockstep task resets, or the first episodes'
                    # fitting transient) is transient and absorbable, while
                    # a non-finite actor gradient means the rollout itself
                    # diverged and must not be silently trained through.
                    for p in critic.parameters():
                        if p.grad is not None:
                            p.grad.nan_to_num_(0.0)
                    # clip_grad_norm_ returns the PRE-clip total norm; keep
                    # it (its mean is logged as critic_grad_norm --
                    # previously discarded, cf. actor_grad_norm).
                    gnorm = torch.nn.utils.clip_grad_norm_(
                        critic.parameters(), self.cfg.grad_norm_clip)
                    opt.step()

                    total_loss += float(loss.item())
                    total_gnorm += float(gnorm.item())
                    steps += 1
                    pass_loss_sum += float(loss.item())
                    pass_steps += 1
            # AHAC-INSPIRED early stop (native fixed-count loop never hits
            # this branch: last_pass_losses is None). Mirrors the official
            # criterion; their check-order differs by at most one pass.
            if last_pass_losses is not None:
                last_pass_losses.append(pass_loss_sum / max(pass_steps, 1))
                if ahac_critic_converged(
                        last_pass_losses, self.cfg.iterative_critic_tol):
                    break
        n = max(steps, 1)
        return total_loss / n, total_gnorm / n

    # ======================================================================
    # STAGE 4: Polyak-update the target critic (paper Algorithm 1)
    # ======================================================================
    def _update_target_critic(self) -> None:
        if not self.target_critics:
            return      # SAPO choice IV: no target networks, no Polyak
        alpha = self.cfg.target_critic_alpha
        with torch.no_grad():
            for critic, target in zip(self.critics, self.target_critics):
                for p, p_tgt in zip(critic.parameters(),
                                    target.parameters()):
                    p_tgt.data.mul_(alpha).add_(p.data, alpha=1.0 - alpha)

    # ======================================================================
    # AHAC-INSPIRED: the adaptive-horizon dual step (NOT native SHAC --
    # runs once per learning episode, only under horizon_mode="adaptive").
    # ======================================================================
    def _update_horizon(self, episode: int) -> Optional[Dict[str, float]]:
        """AHAC delta 1: adapt the window length H via the Lagrangian dual
        on contact stiffness (paper Eq. 9-10; official ahac.py lines --
        see ahac_dual_update for the exact transcription). Called at the
        END of each learning episode, so the cfs_buf still holds THIS
        episode's training window (the gradient-variance diagnostic's
        extra windows, when enabled, ran BEFORE it).

        Returns the logging dict {horizon_H, lambd_mean, cfs_mean}, or
        None in native fixed-horizon mode (the caller then logs Nones so
        the CSV schema is identical across arms)."""
        if not self.adaptive_horizon:
            return None
        cfs_mean = self.cfs_buf[:self.h].mean(dim=1)     # per-step [h]
        # Dual step size: linear decay like the network LRs (official
        # lr_schedule: linear -> (1e-5 - lambd_lr) * frac + lambd_lr).
        frac = min(1.0, (episode - 1) / max(self.cfg.num_episodes, 1))
        lr = ((self.cfg.lambd_lr_final - self.cfg.lambd_lr) * frac
              + self.cfg.lambd_lr)
        self.lambd, self.horizon_float = ahac_dual_update(
            self.lambd, self.horizon_float, cfs_mean,
            self.cfg.contact_threshold, lr,
            self.cfg.steps_min, self.cfg.steps_max)
        self.h = round(self.horizon_float)
        return {"horizon_H": self.horizon_float,
                "lambd_mean": float(self.lambd.mean()),
                "cfs_mean": float(cfs_mean.mean())}

    # ======================================================================
    # Main training loop
    # ======================================================================
    def train(self, resume: Optional[str] = None) -> Dict[str, float]:
        """Run the full training loop (paper Algorithm 1).

        resume: optional path to a checkpoint saved by this class -- restores
        the actor/critic weights, optimizer states, observation statistics,
        dual-tuned alpha, episode counter, env-step count and best-eval
        tracker, then continues from the NEXT episode (LR schedule included,
        since it is keyed on the absolute episode number). NOT restored: the
        torch RNG stream and the simulator's physical state -- a resumed run
        is statistically, not bit-wise, identical to an uninterrupted one.
        """
        if resume is not None:
            self.load(resume)

        best_eval = self._best_eval
        for episode in range(self._episode + 1,
                             self.cfg.num_episodes + 1):
            self._episode = episode
            lr_actor, lr_critic = self._set_lrs(episode - 1)

            # ---- gradient-estimator variance (diagnostic) ----
            # Measured BEFORE this episode's rollout so the k extra
            # windows and the episode's own training window share the
            # SAME theta: the row's actor_grad_norm (one training draw)
            # and actor_grad_var (the estimator's variance at that
            # policy) then describe the same policy.
            gv = (self._measure_grad_variance(self.run_cfg.grad_var_samples)
                  if self.run_cfg.grad_var_interval > 0
                  and episode % self.run_cfg.grad_var_interval == 0
                  else None)

            stats = self._rollout_and_update_actor(
                update_actor=(episode > self.cfg.critic_warmup))  # st. 1-2
            if self.use_critic:
                (stats["critic_loss"],
                 stats["critic_grad_norm"]) = self._update_critic()  # st. 3
                self._update_target_critic()                   # stage 4
            else:
                # Ablation (--no-critic): no critic stages; log empty
                # cells so the CSV schema is identical across arms.
                stats["critic_loss"] = None
                stats["critic_grad_norm"] = None

            # ---- AHAC-INSPIRED: adaptive-horizon dual step ----
            hor = self._update_horizon(episode)
            if hor is not None:
                stats.update(hor)
            else:
                # Native fixed-horizon mode: empty cells keep the CSV
                # schema identical across arms (same convention as the
                # eval/grad-var columns).
                stats["horizon_H"] = None
                stats["lambd_mean"] = None
                stats["cfs_mean"] = None

            # Sparse diagnostic columns (None between measurements, like
            # the eval columns between evals). Always present so every row
            # passes the same key set (see RunLogger.log).
            stats["actor_grad_var"] = gv["actor_grad_var"] if gv else None
            stats["actor_grad_mean_norm"] = (gv["actor_grad_mean_norm"]
                                             if gv else None)

            # ---- periodic evaluation on a separate env (deterministic) ----
            eval_mean = eval_std = None
            if (self.run_cfg.eval_interval > 0
                    and episode % self.run_cfg.eval_interval == 0):
                eval_mean, eval_std = self.evaluate(
                    self.run_cfg.eval_num_episodes)

            # ---- logging row ----
            row = {
                "episode": episode,
                "wall_time_s": self.logger.elapsed() if self.logger else 0.0,
                "env_steps": self.total_env_steps,
                "lr_actor": lr_actor,
                "lr_critic": lr_critic,
                **stats,
                "train_return_mean": (_mean(self._completed_returns)
                                      if self._completed_returns else None),
                "train_return_std": (_std(self._completed_returns)
                                     if self._completed_returns else None),
                "eval_return_mean": eval_mean,
                "eval_return_std": eval_std,
            }
            if self.logger is not None:
                self.logger.log(row)
            self._print_row(row)
            self._completed_returns = []

            # ---- checkpointing ----
            if eval_mean is not None and eval_mean > best_eval:
                best_eval = eval_mean
                self._best_eval = best_eval
                self.save(suffix="best_policy")
            if (self.run_cfg.save_interval > 0
                    and episode % self.run_cfg.save_interval == 0):
                self.save(suffix=f"policy_ep{episode}")
            if episode == self.cfg.num_episodes:
                self.save(suffix="final_policy")

        return {"best_eval_return": best_eval}

    # ======================================================================
    # Evaluation: full task episodes with the deterministic policy
    # (mu, no noise), on a fresh environment so training state is untouched.
    # ======================================================================
    def evaluate(self, num_episodes: int) -> Tuple[float, float]:
        """Deterministic-eval mean/std over `num_episodes` episodes.

        Runs under `torch.random.fork_rng`: the eval env's initial-state
        draws would otherwise consume the GLOBAL torch RNG that also feeds
        the training rollouts, so changing --eval-interval or
        --eval-episodes would silently change the training trajectory
        itself. Forking saves/restores the RNG state around the eval,
        making evaluation transparent to training (all RNG streams,
        CPU and CUDA).
        """
        devices = [self.device] if self.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            return self._evaluate(num_episodes)

    @torch.no_grad()
    def _evaluate(self, num_episodes: int) -> Tuple[float, float]:
        if (not hasattr(self, "_eval_env")
                or self._eval_env.num_envs != num_episodes):
            self._eval_env = make_env(self.env_name, self.env_cfg,
                                      self.device, num_envs=num_episodes)
        env = self._eval_env
        obs = env.reset()
        returns = torch.zeros(env.num_envs, device=self.device)
        lengths = torch.zeros(env.num_envs, device=self.device)
        done_returns: List[float] = []

        # Run until every env has completed exactly one task episode.
        # (Episodes in the current envs always last H steps; the done
        # branch is for generality with future early-termination
        # environments.)
        while len(done_returns) < env.num_envs:
            if self.obs_rms is not None:
                obs = self.obs_rms.normalize(obs)
            action = torch.tanh(self.actor(obs, deterministic=True))
            obs, rew, done, _ = env.step(action)
            returns += rew
            lengths += 1
            done_ids = done.nonzero(as_tuple=False).squeeze(-1)
            if bool(done.any()):
                done_returns.extend(returns[done_ids].tolist())
                returns[done_ids] = 0.0

        return _mean(done_returns), _std(done_returns)

    # ======================================================================
    # Checkpointing
    # ======================================================================
    def save(self, suffix: str = "policy") -> str:
        assert self.logger is not None, "saving requires a logger/logdir"
        path = f"{self.logger.logdir}/{suffix}.pt"
        # Ensemble checkpoints (critic_ensemble > 1) store LISTS under
        # "critics"/"target_critics"; single-critic checkpoints keep the
        # historical "critic"/"target_critic" keys so every old consumer
        # (and old checkpoints) keep working unchanged.
        ensemble = len(self.critics) > 1
        torch.save({
            "actor": self.actor.state_dict(),
            **({"critics": [c.state_dict() for c in self.critics],
                "target_critics": [t.state_dict()
                                   for t in self.target_critics],
                "critic_opts": [o.state_dict() for o in self.critic_opts]}
               if ensemble else
               {"critic": (self.critic.state_dict()
                           if self.use_critic else None),
                "target_critic": (self.target_critic.state_dict()
                                  if self.use_critic else None),
                "critic_opt": (self.critic_opt.state_dict()
                               if self.use_critic else None)}),
            "actor_opt": self.actor_opt.state_dict(),
            "obs_rms": (self.obs_rms.state_dict()
                        if self.obs_rms is not None else None),
            "env_cfg": self.env_cfg.__dict__,
            "env_name": self.env_name,
            "algo_cfg": self.cfg.__dict__,
            # The dual-tuned temperature (None when the max-entropy
            # objective is off). Stored separately from algo_cfg so a
            # non-entropy trainer can load the same checkpoint and simply
            # ignore it.
            "entropy_alpha": (float(self.entropy_alpha)
                              if self.max_entropy else None),
            # AHAC-INSPIRED adaptive-horizon loop state (None in native
            # fixed-horizon mode): H and the per-step dual variables, so
            # --resume continues the dual where it left off.
            "ahac_horizon": (float(self.horizon_float)
                             if self.adaptive_horizon else None),
            "ahac_lambd": (self.lambd.tolist()
                           if self.adaptive_horizon else None),
            # ---- training-loop state (for --resume) ----
            "episode": self._episode,
            "total_env_steps": self.total_env_steps,
            "best_eval_return": (None if self._best_eval == -float("inf")
                                 else float(self._best_eval)),
        }, path)
        return path

    def load(self, path: str) -> None:
        """Restore weights, optimizer states, observation statistics, the
        dual-tuned alpha, and the training-loop state (episode counter,
        env-step count, best-eval tracker) written by save(). Older
        checkpoints without the loop-state keys restore to a fresh loop.
        NOT restored: the torch RNG stream and the simulator's physical
        state (see train()'s resume note)."""
        env_name, _, ckpt = read_checkpoint(path, self.device)
        if env_name != self.env_name:
            raise ValueError(
                f"checkpoint env '{env_name}' does not match this "
                f"trainer's env '{self.env_name}'")
        # The activation is an architecture choice baked into the weights'
        # TRAINING behavior (shapes match regardless, so this would fail
        # silently as a subtly different network) -- refuse the mismatch.
        ckpt_act = ckpt["algo_cfg"].get("activation", "elu")
        if ckpt_act != self.cfg.activation:
            raise ValueError(
                f"checkpoint was trained with activation='{ckpt_act}', "
                f"this trainer uses '{self.cfg.activation}'")
        self.actor.load_state_dict(ckpt["actor"])
        self.actor_opt.load_state_dict(ckpt["actor_opt"])
        if self.use_critic:
            if "critics" in ckpt:          # ensemble checkpoint
                for c, sd in zip(self.critics, ckpt["critics"]):
                    c.load_state_dict(sd)
                for t, sd in zip(self.target_critics,
                                 ckpt["target_critics"]):
                    t.load_state_dict(sd)
                for o, sd in zip(self.critic_opts, ckpt["critic_opts"]):
                    o.load_state_dict(sd)
            else:                          # single-critic checkpoint
                self.critics[0].load_state_dict(ckpt["critic"])
                if (self.target_critics
                        and ckpt.get("target_critic") is not None):
                    self.target_critics[0].load_state_dict(
                        ckpt["target_critic"])
                self.critic_opts[0].load_state_dict(ckpt["critic_opt"])
        if self.obs_rms is not None and ckpt["obs_rms"] is not None:
            self.obs_rms.load_state_dict(ckpt["obs_rms"])
        # Restore the dual-tuned temperature so a resumed max-entropy run
        # continues the schedule where it left off (older/other-mode
        # checkpoints simply have no key, and non-entropy trainers ignore
        # the field entirely).
        if (self.max_entropy and ckpt.get("entropy_alpha") is not None):
            self.entropy_alpha = float(ckpt["entropy_alpha"])
        # AHAC-INSPIRED: restore H and the dual variables (absent in
        # native fixed-horizon checkpoints and pre-AHAC saves -> fresh).
        if (self.adaptive_horizon
                and ckpt.get("ahac_horizon") is not None):
            self.horizon_float = float(ckpt["ahac_horizon"])
            self.lambd = torch.tensor(ckpt["ahac_lambd"],
                                      dtype=torch.float32,
                                      device=self.device)
            self.h = round(self.horizon_float)
        # Training-loop state (absent in pre-resume checkpoints -> fresh).
        self.total_env_steps = int(ckpt.get("total_env_steps", 0))
        self._episode = int(ckpt.get("episode", 0))
        best = ckpt.get("best_eval_return")
        self._best_eval = (-float("inf") if best is None else float(best))

    # ======================================================================
    # Console output
    # ======================================================================
    def _print_row(self, row: Dict[str, float]) -> None:
        cl = (f"{row['critic_loss']:8.4f}" if row["critic_loss"] is not None
              else "     n/a")
        msg = (f"ep {row['episode']:4d}/{self.cfg.num_episodes} "
               f"| steps {row['env_steps']:7d} "
               f"| actor_loss {row['actor_loss']:9.2f} "
               f"| critic_loss {cl} "
               f"| std {row['policy_std']:.3f} "
               f"| lr {row['lr_actor']:.2e}")
        if row.get("horizon_H") is not None:
            # AHAC-INSPIRED adaptive horizon: window length, dual mean and
            # mean stiffness (the Prediction-0 evidence in AHAC.md).
            msg += (f" | H {row['horizon_H']:5.1f}"
                    f" | lam {row['lambd_mean']:+8.2f}"
                    f" | cfs {row['cfs_mean']:6.2f}")
        if row["train_return_mean"] is not None:
            msg += f" | train_ret {row['train_return_mean']:8.1f}"
        if row.get("entropy_alpha") is not None:
            msg += (f" | a* {row['entropy_alpha']:.3f}"
                    f" | H {row['policy_entropy']:+.2f}")
        if row["eval_return_mean"] is not None:
            msg += (f" | EVAL ret {row['eval_return_mean']:8.1f}"
                    f" +/- {row['eval_return_std']:.1f}")
        if row.get("actor_grad_var") is not None:
            msg += (f" | gvar {row['actor_grad_var']:.3g}"
                    f" | gmean {row['actor_grad_mean_norm']:.3g}")
        print(msg, flush=True)


# ---------------------------------------------------------------------------
# Tiny statistics helpers (numpy-free)
# NOTE: population convention (divide by n) -- these describe the COMPLETED
# episodes at hand. Across-seed summaries in scripts/plot_ablation.py use
# the sample convention (ddof=1) instead: there the seeds are a sample of a
# seed distribution whose spread is being ESTIMATED. Different estimands.
# ---------------------------------------------------------------------------
def _mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def _std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0
    mu = _mean(xs)
    return (sum((x - mu) ** 2 for x in xs) / len(xs)) ** 0.5
