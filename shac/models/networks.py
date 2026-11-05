"""
Actor and critic networks for SHAC.

Paper A.2.2 (implementation details of our method):
    "We use multilayer perceptron (MLP) as the policy for our method. We use
     ELU as our activation function for the hidden layers and apply layer
     normalization to each hidden layer."

Network sizes for CartPole (paper Table 1):
    policy network: [64, 64]
    value network: [64, 64]

THE ACTOR is stochastic (paper Section 3.3):
    a_t ~ N( mu_theta(s_t), sigma_theta(s_t) )
and SHAC samples it with the REPARAMETERIZATION trick so that the gradient
of the sampled action w.r.t. the policy parameters exists:

    a = mu(s) + sigma * epsilon,    epsilon ~ N(0, I)

The official implementation parameterizes sigma as a single global learnable
log-std vector (one entry per action dimension, init -1 -> sigma ~= 0.37)
rather than a state-dependent network output; we follow the official code
(the paper's sigma_theta(s) notation is looser than what was released).

SHACConfig.actor_state_std = True switches to that SAC-style state-dependent
parameterization: the actor head emits [mu | log_std(s)] (2 x action_dim),
log_std clamped to [LOG_STD_MIN, LOG_STD_MAX] as in SAC. The log-std half
of the head is zero-initialized with bias logstd_init, so both variants
start from the IDENTICAL sigma(s) = exp(logstd_init) and the A/B between
them is clean (only the ability to learn state dependence differs).

The trainer additionally squashes the sampled action with tanh before
applying it (official code: `env.step(torch.tanh(actions))`), which bounds
the applied force; gradients flow through tanh.
"""

from typing import Tuple

import math

import torch
import torch.nn as nn


# SAC-style bounds on the state-dependent log-std head (Haarnoja et al.,
# "Soft Actor-Critic" clamps log_std to [-20, 2]).
LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0

# Guard for log(1 - tanh^2(u)): the squashed-Gaussian entropy diverges as
# the sampled action saturates toward +/-1 (SAC adds the same epsilon).
TANH_ENTROPY_EPS = 1e-6


def tanh_gaussian_entropy(u: torch.Tensor,
                          sigma: torch.Tensor) -> torch.Tensor:
    """Per-dim entropy of the squashed Gaussian a = tanh(u),
    u ~ N(mu, sigma^2), estimated at the SAMPLED u (SAC's estimator).

    Change of variables a = tanh(u) gives
    log pi(a|s) = log N(u|mu,sigma) - log(1 - tanh^2(u)), so

        H[a|s] = E[-log pi] = log sigma + 0.5*log(2*pi*e)
                             + E[log(1 - tanh^2(u))]

    The tanh term is <= 0: SQUASHING REDUCES entropy (an action driven
    into saturation is nearly deterministic, hence low-entropy), and its
    gradient pushes mu back from the boundary. The expectation is
    approximated by its value at the reparameterized sample (the E[eps^2]
    part of the exact -log pi is replaced by its mean 1: it has zero mean
    and zero theta-gradient in expectation, so this estimator is
    unbiased with LOWER variance than SAC's per-sample -log pi).
    Differentiable in mu (the tanh correction) and in sigma (log sigma)
    -- both inside the SHAC window's computation graph, so the entropy
    term trains the policy through the same FOBG path as the reward
    (SAPO Eq. 18-19).

    Args:
      u:     [batch, action_dim]  the SAMPLED pre-tanh actions.
      sigma: [batch, action_dim]   policy std at the same states.
    Returns:
      [batch, action_dim] per-dimension entropies.
    """
    gauss = torch.log(sigma) + 0.5 * math.log(2.0 * math.pi * math.e)
    correction = torch.log(1.0 - torch.tanh(u) ** 2 + TANH_ENTROPY_EPS)
    return gauss + correction


# ---------------------------------------------------------------------------
# MLP builder
# ---------------------------------------------------------------------------
def build_mlp(input_dim: int, output_dim: int, units: Tuple[int, ...],
              activation: str = "elu", orthogonal_init: bool = True) -> nn.Sequential:
    """Linear stack: [Linear -> activation -> LayerNorm] per hidden layer,
    then a bare Linear head.

    `orthogonal_init=True` initializes hidden linear layers with the scaled
    orthogonal scheme (gain sqrt(2), zero bias) as in the official code's
    critic; the released *stochastic actor* actually left its linears at
    PyTorch's default init -- we use orthogonal init for both here, which is
    the more standard choice and trains equivalently on this task.
    """
    act_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    dims = [input_dim] + list(units) + [output_dim]
    layers = []
    for i in range(len(dims) - 1):
        linear = nn.Linear(dims[i], dims[i + 1])
        is_hidden = i < len(dims) - 2
        if orthogonal_init and is_hidden:
            nn.init.orthogonal_(linear.weight, gain=2.0 ** 0.5)
            nn.init.zeros_(linear.bias)
        layers.append(linear)
        if is_hidden:  # activation + LayerNorm on every hidden layer (paper)
            layers.append(act_fn())
            layers.append(nn.LayerNorm(dims[i + 1]))
    return nn.Sequential(*layers)


# ---------------------------------------------------------------------------
# Actor
# ---------------------------------------------------------------------------
class GaussianActor(nn.Module):
    """Stochastic Gaussian policy.

    forward(obs)             -> reparameterized sample a ~ N(mu(s), sigma(s))
    forward(obs, deterministic=True) -> mu(s)     (used for evaluation)
    mu_sigma(obs)            -> (mu(s), sigma(s)) tensors for analysis

    Two parameterizations of sigma (SHACConfig.actor_state_std):
      * global (default, official SHAC): one learnable log-std parameter
        per action dimension -- sigma is the same at every state.
      * state (SAC-style): the MLP head emits [mu | log_std(s)]; log_std
        is clamped to [LOG_STD_MIN, LOG_STD_MAX]. The log-std half of the
        head is zero-initialized (bias = logstd_init), so training STARTS
        at sigma(s) == exp(logstd_init) for every s and only learns state
        dependence if the gradient asks for it.
    """

    def __init__(self, obs_dim: int, action_dim: int, units: Tuple[int, ...],
                 logstd_init: float = -1.0, state_std: bool = False):
        super().__init__()
        self.action_dim = action_dim
        self.state_std = state_std
        # policy_std accumulator for the state mode (see std()); kept in
        # both modes so reset_std() is always safe to call.
        self._std_acc, self._std_n = 0.0, 0
        if state_std:
            # One MLP with a SAC-style [mu | log_std(s)] head.
            self.mu_net = build_mlp(obs_dim, 2 * action_dim, units)
            head = [m for m in self.mu_net.modules()
                    if isinstance(m, nn.Linear)][-1]
            with torch.no_grad():
                head.weight[action_dim:].zero_()
                head.bias[action_dim:] = float(logstd_init)
        else:
            self.mu_net = build_mlp(obs_dim, action_dim, units)
            # One learnable log-std per action dimension (official ActorStochasticMLP).
            self.logstd = nn.Parameter(
                torch.full((action_dim,), float(logstd_init)))

    def mu_sigma(self, obs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Mean and std of the policy at obs: both [batch, action_dim]."""
        if not self.state_std:
            mu = self.mu_net(obs)
            sigma = self.logstd.exp().expand_as(mu)
            return mu, sigma
        mu, log_std = self.mu_net(obs).split(self.action_dim, dim=-1)
        sigma = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX).exp()
        return mu, sigma

    def _sample_core(self, obs: torch.Tensor) -> Tuple[torch.Tensor,
                                                        torch.Tensor,
                                                        torch.Tensor]:
        """The single reparameterized-sampling path shared by forward()
        and sample_with_entropy() (one randn draw, one std-accumulator
        update in state mode) -- so switching the trainer to the
        entropy objective does NOT perturb the RNG stream."""
        mu, sigma = self.mu_sigma(obs)
        # Reparameterization: the noise is external to the parameters, so
        # d(sample)/d(theta) = d(mu)/d(theta) + epsilon * d(sigma)/d(theta).
        eps = torch.randn_like(mu)
        if self.state_std:
            # policy_std diagnostics: accumulate the mean sigma(s) over
            # the stochastic forward calls since the last reset_std();
            # std() reads and resets the accumulator.
            with torch.no_grad():
                self._std_acc += float(sigma.mean())
                self._std_n += 1
        return mu + sigma * eps, mu, sigma

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        if deterministic:
            return self.mu_sigma(obs)[0]
        return self._sample_core(obs)[0]

    def sample_with_entropy(self, obs: torch.Tensor) -> Tuple[torch.Tensor,
                                                               torch.Tensor]:
        """Reparameterized sample PLUS its per-dim tanh-Gaussian entropy.

        Returns (u, h): the pre-tanh action u ~ N(mu(s), sigma(s)) and
        h = the squashed-Gaussian entropy estimate at that sample
        (see tanh_gaussian_entropy). Used by the max-entropy trainer
        (SHACConfig.max_entropy) to add SAPO's alpha * entropy term to
        the window return; identical RNG consumption to forward().
        """
        u, _, sigma = self._sample_core(obs)
        return u, tanh_gaussian_entropy(u, sigma)

    def reset_std(self) -> None:
        """Start a fresh policy_std accumulation window (state mode)."""
        self._std_acc, self._std_n = 0.0, 0

    @torch.no_grad()
    def std(self) -> float:
        """Current action std (logged for diagnostics).

        Global mode: the learnable sigma. State mode: the mean sigma(s)
        over the stochastic forward calls since the last reset_std()/std()
        call -- i.e. the average exploration noise over the states of the
        episode just rolled out (SHACTrainer resets the accumulator at the
        start of every window, so probe windows never pollute it).
        Deterministic eval calls are not accumulated."""
        if not self.state_std:
            return float(self.logstd.exp().mean().item())
        if self._std_n == 0:
            return 0.0
        m = self._std_acc / self._std_n
        self.reset_std()
        return m


# ---------------------------------------------------------------------------
# Critic
# ---------------------------------------------------------------------------
class Critic(nn.Module):
    """State-value function V_phi(s) -> scalar."""

    def __init__(self, obs_dim: int, units: Tuple[int, ...]):
        super().__init__()
        self.v_net = build_mlp(obs_dim, 1, units)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """Returns V(s) with shape [batch, 1] (squeeze in the caller)."""
        return self.v_net(obs)
