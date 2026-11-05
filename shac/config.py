"""
Configuration for the SHAC CartPole Swing-Up experiment.

Every default below is taken from either the paper (tables referenced in the
comments) or the authors' released configuration file for this exact task:

    NVlabs/DiffRL: examples/cfg/shac/cartpole_swing_up.yaml

so that training here reproduces the paper's CartPole experiment as closely
as possible. Where the paper and the released code differ (or the paper is
silent), the comment explains the choice.
"""

from dataclasses import dataclass, field
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Environment: CartPole Swing-Up (paper Appendix A.3.1)
# ---------------------------------------------------------------------------
@dataclass
class EnvConfig:
    """CartPole Swing-Up environment parameters.

    Observation (paper Table 6), 5 values:
        [x, x_dot, sin(theta), cos(theta), theta_dot]

    Action (1 value):
        torque applied to the cart's prismatic joint. The raw action is
        squashed to (-1, 1) (tanh in the trainer, clip here as a safety net)
        and multiplied by `action_strength` to get the applied force in
        Newtons.

    Reward (paper Eq. 13):
        R = -theta^2 - 0.1 * theta_dot^2 - 0.05 * x^2 - 0.1 * x_dot^2
        (always <= 0; a perfectly balanced pole earns ~0 per step)
    """

    # --- parallel environments -------------------------------------------
    num_envs: int = 64
    # Number of environments simulated in parallel (paper Table 4: N = 64).
    # This is "N" in the paper's equations.

    episode_length: int = 240
    # Task horizon H (paper A.3.1: H = 240 steps). The environment resets
    # after this many steps. Note: this is NOT the SHAC short horizon h.

    # --- time discretization ----------------------------------------------
    dt: float = 1.0 / 60.0
    # Duration of one policy step in seconds (official env: 1/60 s, i.e.
    # the policy runs at 60 Hz).
    sim_substeps: int = 4
    # Physics sub-integration steps per policy step (official env: 4).
    # Each substep integrates dt/4 = 1/240 s with semi-implicit Euler
    # (paper Section 3.1: "semi-implicit Euler integration").

    # --- physical parameters ------------------------------------------------
    # These come from the official dflex cartpole URDF asset
    # (envs/assets/cartpole.urdf in NVlabs/DiffRL):
    #   cart: mass 1.0 kg (slides on a prismatic x-axis joint)
    #   pole: mass 10.0 kg, center of mass 0.5 m above the pivot,
    #         rotational inertia 1.0 kg m^2 about its own COM (y axis)
    #         -> inertia about the pivot: I = 1.0 + 10 * 0.5^2 = 3.5
    cart_mass: float = 1.0
    pole_mass: float = 10.0
    pole_com_distance: float = 0.5     # l: distance pivot -> pole COM [m]
    pole_inertia_com: float = 1.0      # I_com: pole inertia about its COM
    gravity: float = 9.81              # m / s^2

    action_strength: float = 1000.0
    # Force scale: applied_force = action_strength * action. Matches the
    # URDF joint effort limit (1000 N), so a saturated action == maximum
    # admissible force.

    # --- soft joint limit on the cart (paper Eq. 4) --------------------------
    cart_limit: float = 4.0            # slider travel limits: [-4, 4] m (URDF)
    limit_stiffness: float = 100.0     # k_limit: penalty stiffness
    limit_damping: float = 1.0         # damping on limit engagement (dflex style)
    # The pole's revolute joint is "continuous" (no limits) in the URDF.
    # The limit force is essentially never active for reasonable policies
    # (the reward already penalizes |x|), but it keeps simulations bounded.

    # --- initial state distribution ------------------------------------------
    stochastic_init: bool = True
    # Official env: every reset starts from the fixed pose (x=0, theta=-pi,
    # i.e. pole hanging straight down) plus uniform noise:
    #   joint positions += pi   * (U[0,1] - 0.5)   -> x in [-pi/2, pi/2],
    #                                                 theta in [-3pi/2, -pi/2]
    #   joint velocities += 0.5 * (U[0,1] - 0.5)  -> each velocity in [-0.25, 0.25]

    early_termination: bool = False
    # CartPole Swing-Up has NO early termination (paper A.3.1), unlike e.g.
    # Hopper/Ant. Environments only reset at the task horizon H=240.

    # --- misc ----------------------------------------------------------------
    dtype: str = "float32"
    # Float precision used inside the simulator. float32 matches the
    # official implementation; tests switch to float64 for tight tolerances.

    def __post_init__(self):
        assert self.num_envs >= 1
        assert self.episode_length > 0


# ---------------------------------------------------------------------------
# SHAC algorithm hyperparameters (paper Tables 4 & 5, CartPole column,
# cross-checked against the released cartpole_swing_up.yaml)
# ---------------------------------------------------------------------------
@dataclass
class SHACConfig:
    """Hyperparameters of the SHAC algorithm itself."""

    # --- short-horizon learning window ---------------------------------------
    short_horizon: int = 32
    # h: number of steps per learning episode (paper Table 4: h = 32).
    # Gradients are backpropagated through this window only, then cut.

    num_episodes: int = 500
    # M: total number of learning episodes (paper Table 4: M = 500 for
    # CartPole; each episode consumes h * N = 32 * 64 = 2048 env steps).

    # --- discounting / value estimation ----------------------------------------
    gamma: float = 0.99        # discount factor (paper Eq. 5/7)
    td_lambda: float = 0.95    # lambda for td-(lambda) value estimation (Eq. 7)

    # --- optimization -----------------------------------------------------------
    actor_learning_rate: float = 1e-2    # paper Table 4 (CartPole)
    critic_learning_rate: float = 1e-3    # paper Table 4 (CartPole)
    lr_final: float = 1e-5
    # Learning rates decay linearly from their initial value to `lr_final`
    # over the course of training (official implementation; the paper only
    # says "learning rates follow a linear decay schedule over episodes").

    adam_betas: Tuple[float, float] = (0.7, 0.95)   # paper Table 4

    grad_norm_clip: float = 1.0
    # Gradient-norm clipping applied to BOTH actor and critic gradients
    # (official config: truncate_grads=True, grad_norm=1.0). The paper does
    # not mention this, but the released implementation uses it.

    # --- critic training --------------------------------------------------------
    critic_iterations: int = 16      # passes over the episode's data (Table 4)
    critic_num_batch: int = 4        # minibatches per iteration (Table 4)
    # Each learning episode therefore performs 16 * 4 = 64 critic gradient
    # steps. The minibatches are CONTIGUOUS temporal slices of the rollout
    # buffer (the official CriticDataset does not shuffle by default):
    # batch 0 = states from steps [0, h/4), batch 1 = steps [h/4, h/2), ...

    # --- target value network ----------------------------------------------------
    target_critic_alpha: float = 0.2
    # Polyak averaging coefficient (paper Algorithm 1, CartPole column of
    # Table 4): V' <- alpha * V' + (1 - alpha) * V after each episode.
    # NOTE: the paper's "general" setting (Table 5) uses 0.995, but CartPole
    # uses the much faster 0.2 (target moves 80% toward the critic each
    # episode). Keep 0.2 for CartPole.

    # --- actor network -------------------------------------------------------
    actor_units: Tuple[int, ...] = (64, 64)   # paper Table 1 (CartPole)
    critic_units: Tuple[int, ...] = (64, 64)  # paper Table 1 (CartPole)
    # Both networks are MLPs with ELU activations and LayerNorm on every
    # hidden layer (paper A.2.1/2.2).

    actor_logstd_init: float = -1.0
    # The stochastic policy is a ~ N(mu(s), sigma) with a GLOBAL learnable
    # log-standard-deviation (one parameter per action dimension), following
    # the official ActorStochasticMLP. The paper writes sigma_theta(s)
    # (state-dependent), but the released implementation uses a single
    # learnable sigma = exp(logstd) initialized to exp(-1) ~= 0.37.

    actor_state_std: bool = False
    # Experimental A/B (--state-std): switch sigma to the SAC-style
    # STATE-DEPENDENT parameterization the paper's notation describes --
    # the actor head emits [mu | log_std(s)], log_std clamped to [-20, 2]
    # as in SAC. The log-std half of the head is zero-initialized with
    # bias actor_logstd_init, so both variants start from the IDENTICAL
    # sigma(s) = exp(actor_logstd_init) and differ only in whether sigma
    # can become state-dependent. NOTE: the SHAC actor loss has no
    # entropy/log-prob term (unlike SAC's objective) -- sigma(s) learns
    # only through the reparameterized noise in the action path, i.e.
    # dL/dsigma(s) = dL/du * eps. policy_std then logs the window-mean
    # of sigma(s). Checkpoints from the two modes are NOT
    # architecture-compatible (load with the same --state-std).

    # --- maximum entropy objective (SAC-style; cf. MAX_ENTROPY.md) -------
    max_entropy: bool = False
    # The SAPO arm (Xing et al., ICLR 2025): augment the actor's window
    # return with alpha * entropy (their Eq. 18-19) and the critic's
    # td-(lambda) targets with the same term (soft values, Eq. 20), with
    # the temperature alpha auto-tuned by SAC's dual (Eq. 21) to hold the
    # policy entropy near entropy_target. This adds the entropy
    # counter-pressure whose ABSENCE collapsed sigma(s) in the state-std
    # A/B (STATE_STD.md). Works with BOTH sigma parameterizations; the
    # architecture is unchanged, so checkpoints load across this flag.

    entropy_target: Optional[float] = None
    # Target policy entropy H-bar for the alpha dual. None -> -dim(A)/2
    # (SAC convention via Ball et al. 2023, also SAPO's choice);
    # -0.5 for CartPole's 1-D action.

    entropy_lr: float = 5e-3
    # Dual-step learning rate for alpha (SAPO's "entropy eta").

    entropy_init_alpha: float = 1.0
    # Initial temperature (SAPO Table 4). alpha >= 0 always (clamped).
    # NOTE: like SAPO, the entropy is normalized to ~[0, 1] by H-bar
    # before entering the actor loss and the critic targets
    # (their design choice II); the dual step uses the RAW entropy.

    # --- ablation ------------------------------------------------------------
    use_critic: bool = True
    # Ablation switch (--no-critic): when False, the actor loss is ONLY the
    # discounted short-horizon window reward -- the pure pathwise-derivative
    # part of paper Eq. 5, without the terminal bootstrap gamma^h * V'(s_h).
    # No critic/target networks, no td-(lambda), no Polyak step. This turns
    # SHAC into discounted short-horizon BPTT, isolating what the critic
    # contributes (long-horizon credit beyond the window; paper Sec. 3.3).

    # --- observation normalization ---------------------------------------------
    obs_rms: bool = True
    # Normalize observations with a running mean/std (paper Section 3.3:
    # "normalizes the state observation by a running mean and standard
    # deviation calculated from the state observations in previous learning
    # episodes"). Within a learning episode the statistics are frozen;
    # they are still updated online so later episodes see them.

    def __post_init__(self):
        assert 0 < self.gamma <= 1.0
        assert 0 < self.td_lambda <= 1.0
        assert self.short_horizon >= 2
        assert self.critic_num_batch >= 1


# ---------------------------------------------------------------------------
# Experiment / training-run settings (not part of the algorithm)
# ---------------------------------------------------------------------------
@dataclass
class RunConfig:
    """Bookkeeping settings for a training run."""

    seed: int = 0
    device: str = "auto"
    # "auto" | "cpu" | "cuda" | "cuda:0" ... The whole implementation is
    # device-agnostic; CPU is plenty fast for CartPole at N=64 (the parallel
    # env is a handful of small tensor ops), GPU pays off at larger N.

    logdir: Optional[str] = None     # None -> runs/cartpole_<date>_<time>
    eval_interval: int = 10         # evaluate every k learning episodes
    eval_num_episodes: int = 16     # episodes per evaluation (on a separate env)
    save_interval: int = 100        # checkpoint every k episodes (+ best/last)

    # --- gradient-estimator diagnostics ----------------------------------
    # One training episode yields exactly ONE draw of the actor gradient
    # g(theta, xi); consecutive rows mix estimator noise with the drift of
    # the true gradient as the policy learns, so the estimator's VARIANCE
    # is not identifiable from the training log alone (see
    # SHACTrainer._measure_grad_variance). Every `grad_var_interval`
    # episodes, k = `grad_var_samples` extra windows are rolled out at the
    # CURRENT theta (no optimizer step) to sample the estimator directly:
    #   actor_grad_var        =  E[||g||^2] - ||E[g]||^2   (trace of the
    #                           empirical gradient covariance)
    #   actor_grad_mean_norm  =  ||E[g]||  (best estimate of the true
    #                           gradient norm at theta)
    # Cost: k extra windows (~1.5 s per measurement on CPU); they advance
    # the environment stream but are not counted as training env steps.
    # This is the quantity behind the paper's short-horizon motivation
    # (Sec. 3.2: BPTT's gradient variance grows with the horizon; h keeps
    # it bounded) -- measured live along training.
    grad_var_interval: int = 25     # measure every k episodes (0 = off)
    grad_var_samples: int = 8       # extra windows per measurement (>= 2)

    def __post_init__(self):
        if self.device == "auto":
            pass                    # resolved later, needs torch import
        assert self.grad_var_interval >= 0
        if self.grad_var_interval > 0:
            assert self.grad_var_samples >= 2, (
                "gradient-variance estimation needs >= 2 windows per "
                "measurement (RunConfig.grad_var_samples)")
