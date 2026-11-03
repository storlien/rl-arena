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

The trainer additionally squashes the sampled action with tanh before
applying it (official code: `env.step(torch.tanh(actions))`), which bounds
the applied force; gradients flow through tanh.
"""

from typing import Tuple

import torch
import torch.nn as nn


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
    """Stochastic Gaussian policy with a global learnable standard deviation.

    forward(obs)             -> reparameterized sample a ~ N(mu(s), sigma)
    forward(obs, deterministic=True) -> mu(s)     (used for evaluation)
    """

    def __init__(self, obs_dim: int, action_dim: int, units: Tuple[int, ...],
                 logstd_init: float = -1.0):
        super().__init__()
        self.mu_net = build_mlp(obs_dim, action_dim, units)
        # One learnable log-std per action dimension (official ActorStochasticMLP).
        self.logstd = nn.Parameter(torch.full((action_dim,), float(logstd_init)))
        self.action_dim = action_dim

    def forward(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        mu = self.mu_net(obs)
        if deterministic:
            return mu
        sigma = self.logstd.exp()
        # Reparameterization: the noise is external to the parameters, so
        # d(sample)/d(theta) = d(mu)/d(theta) + epsilon * d(sigma)/d(theta).
        eps = torch.randn_like(mu)
        return mu + sigma * eps

    @torch.no_grad()
    def std(self) -> float:
        """Current global action std (logged for diagnostics)."""
        return float(self.logstd.exp().mean().item())


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
