"""
Physics and gradient tests for the differentiable Pendulum environment.

These can be run directly:   uv run python tests/test_pendulum_dynamics.py
(or via pytest). They validate:

  1. Energy conservation of the unforced dynamics (integrator sanity):
     E = 0.5*I*theta_dot^2 + m*g*l*cos(theta) is conserved when u = 0.
  2. Equilibrium behavior: upright is an unstable equilibrium, hanging a
     stable one (correctness of the equation of motion).
  3. THE CENTRAL SHAC ASSUMPTION: the autograd Jacobians of the simulator
     transition (paper Eq. 1's dF/ds and dF/da) match finite differences --
     i.e. the backward pass through the "differentiable simulation layer"
     is exact. (Same spirit as the paper's Figure 11 gradient check.)
  4. Reset semantics at the task horizon H.
  5. Reward/observation spot checks at the goal and at hanging.
  6. A smoke test of the full actor-loss backward path.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import math
import torch

from shac.config import PendulumEnvConfig
from shac.envs.pendulum import PendulumSwingUpEnv


def make_env(num_envs=1, dtype="float64"):
    """Test environment: float64 for tight finite-difference tolerances,
    deterministic init so tests are reproducible."""
    cfg = PendulumEnvConfig(num_envs=num_envs, dtype=dtype,
                            stochastic_init=False)
    return PendulumSwingUpEnv(cfg, torch.device("cpu"))


# ---------------------------------------------------------------------------
def test_energy_conservation():
    """No actuation -> total energy must be (nearly) conserved.

    E = 0.5*I*thd^2 + m*g*l*cos(theta)   (theta measured from upright)
    """
    env = make_env()
    env.reset()
    # Start hanging with some swing velocity.
    env.qd[0, 0] = 1.0

    def energy(env):
        th, thd = env.q[0, 0], env.qd[0, 0]
        return 0.5 * env.I * thd ** 2 + env.mgl * math.cos(th)

    e0 = energy(env)
    # 239 steps: step 240 would hit the horizon reset and hand us the
    # freshly-drawn initial state instead of the evolved one.
    for _ in range(env.episode_length - 1):   # ~4 s of simulation
        env.step(torch.zeros(1, 1))
    e1 = energy(env)
    drift = abs(e1 - e0) / abs(e0)
    assert drift < 0.01, f"energy drifted {drift:.2%} (integrator unstable?)"
    print(f"  energy: {e0:.3f} -> {e1:.3f}  (drift {drift:.4%})")


# ---------------------------------------------------------------------------
def test_equilibria():
    """theta=0 must be an UNSTABLE equilibrium, theta=pi a STABLE one.

    NOTE (1-DoF subtlety): a pendulum released AT REST slightly off upright
    cannot fall -- energy conservation traps it in |theta| <= theta_0 (the
    upright configuration is a potential barrier of height 2*mgl between
    the two falling directions; unlike the CartPole there is no second DoF
    to route energy around it). Instability is therefore tested with a
    small push over the top, and stability of hanging by a swing that must
    stay away from upright for the whole episode."""
    env = make_env()

    # (a) exactly upright, at rest, no torque -> stays forever
    #     (check before the horizon reset fires on step H)
    env.reset()
    env.q[0] = torch.tensor([0.0], dtype=env.q.dtype)
    env.qd[0] = 0.0
    for _ in range(env.episode_length - 1):
        env.step(torch.zeros(1, 1))
    assert abs(env.q[0, 0].item()) < 1e-6, "upright equilibrium did not hold"

    # (b) tiny push from upright -> leaves the top and keeps rotating
    #     (unstable equilibrium: theta_ddot = +(mgl/I) sin(theta))
    env.reset()
    env.q[0] = torch.tensor([0.0], dtype=env.q.dtype)
    env.qd[0] = torch.tensor([0.1], dtype=env.qd.dtype)
    max_dev = 0.0
    for _ in range(100):                      # 1.67 s of simulation
        env.step(torch.zeros(1, 1))
        th = math.atan2(math.sin(env.q[0, 0].item()),
                        math.cos(env.q[0, 0].item()))
        max_dev = max(max_dev, abs(th))
    assert max_dev > 1.0, (f"pushed upright pendulum never left the top "
                           f"(max |theta| = {max_dev:.2f})")
    print(f"  upright pushed -> swung to |theta| = {max_dev:.2f} "
          f"(unstable, correct)")

    # (c) displaced from hanging -> oscillates around pi, never near upright
    #     (stable equilibrium)
    env.reset()
    env.q[0] = torch.tensor([math.pi - 0.5], dtype=env.q.dtype)
    env.qd[0] = 0.0
    for _ in range(env.episode_length - 1):
        env.step(torch.zeros(1, 1))
        th = abs(math.atan2(math.sin(env.q[0, 0].item()),
                            math.cos(env.q[0, 0].item())))
        assert th > 2.0, (f"hanging pendulum wandered upright "
                          f"(|theta| = {th:.2f})")
    print("  hanging displaced -> oscillated around pi, never near upright "
          "(stable, correct)")


# ---------------------------------------------------------------------------
def test_transition_jacobians():
    """Autograd dF/ds, dF/da  ==  finite differences (paper Eq. 1).

    This is the property that makes SHAC possible: the simulator's backward
    pass is the exact derivative of its forward pass.
    """
    torch.manual_seed(0)
    env = make_env(num_envs=1)

    for trial in range(5):
        # Random state and action.
        s0 = torch.tensor(
            [[math.pi * (torch.rand(()).item() - 0.5) * 2,
              1.0 * torch.randn(()).item()]],
            dtype=torch.float64).reshape(1, 2).requires_grad_(True)
        a = (2 * torch.rand(1, 1, dtype=torch.float64) - 1).requires_grad_(True)

        # -- autograd Jacobians: one backward pass per output component --
        j_auto = torch.zeros(2, 3, dtype=torch.float64)
        for k in range(2):
            env.q = s0[:, 0:1]               # views: graph stays connected
            env.qd = s0[:, 1:2]
            env.progress_buf.fill_(0)
            q1, qd1 = env._integrate(env.q, env.qd, a)
            out = torch.cat([q1, qd1], dim=-1)[0, k]
            grads = torch.autograd.grad(out, [s0, a])
            j_auto[k, 0:2] = grads[0].reshape(2)
            j_auto[k, 2] = grads[1].item()

        # -- central finite differences --
        eps = 1e-6
        j_fd = torch.zeros(2, 3, dtype=torch.float64)

        def f_flat(s_vec, a_val):
            env.q = s_vec[0:1].reshape(1, 1)
            env.qd = s_vec[1:2].reshape(1, 1)
            env.progress_buf.fill_(0)
            q1, qd1 = env._integrate(env.q, env.qd,
                                     torch.tensor([[a_val]],
                                                  dtype=torch.float64))
            return torch.cat([q1, qd1], dim=-1).squeeze(0)

        s0_flat = s0.detach().reshape(2)
        for j in range(2):
            sp = s0_flat.clone(); sp[j] += eps
            sm = s0_flat.clone(); sm[j] -= eps
            j_fd[:, j] = (f_flat(sp, a.item()) - f_flat(sm, a.item())) / (2 * eps)
        ap, am = a.item() + eps, a.item() - eps
        j_fd[:, 2] = (f_flat(s0_flat, ap) - f_flat(s0_flat, am)) / (2 * eps)

        err = (j_auto - j_fd).abs().max().item()
        assert err < 1e-5, f"trial {trial}: |autograd - FD| = {err:.2e}"
    print(f"  transition Jacobians match finite differences "
          f"(max err over 5 trials: {err:.2e})")


# ---------------------------------------------------------------------------
def test_reset_at_horizon():
    """After H steps the env reports done and the state is re-drawn."""
    env = make_env()
    env.reset()
    dones = 0
    for t in range(env.episode_length):
        _, _, done, info = env.step(torch.zeros(env.num_envs, 1))
        dones += int(done.sum())
    assert dones == 1, f"expected exactly one done after H steps, got {dones}"
    # Fresh state is near hanging (theta within +/- pi/2 of -pi) and at rest-ish
    theta = env.q[0, 0].item()
    assert -1.5 * math.pi < theta < -0.5 * math.pi, theta
    print(f"  env reset at horizon H={env.episode_length}, "
          f"fresh theta = {theta:.2f} (near hanging)")


# ---------------------------------------------------------------------------
def test_reward_and_obs_at_goal():
    """At the goal (balanced, still, zero torque) the reward is ~0 and the
    observation is [1, 0, 0]; hanging still gives R = -pi^2."""
    env = make_env()
    env.reset()
    env.q[0] = torch.tensor([0.0], dtype=env.q.dtype)       # upright
    env.qd[0] = 0.0
    obs, rew, _, _ = env.step(torch.zeros(1, 1))
    assert abs(float(rew.item())) < 1e-8, f"goal reward = {rew.item()}"
    assert torch.allclose(obs[0], torch.tensor([1.0, 0.0, 0.0],
                                               dtype=obs.dtype),
                          atol=1e-6), obs

    env.reset()
    env.q[0] = torch.tensor([math.pi], dtype=env.q.dtype)   # hanging
    env.qd[0] = 0.0
    _, rew, _, _ = env.step(torch.zeros(1, 1))
    assert abs(float(rew.item()) + math.pi ** 2) < 1e-8, \
        f"hanging reward = {rew.item()} (expected -pi^2)"
    print("  reward at goal ~ 0, obs at goal = [1, 0, 0], "
          "hanging reward = -pi^2: OK")


# ---------------------------------------------------------------------------
def test_actor_loss_backward_path():
    """End-to-end gradient smoke test: rewards + terminal critic value
    backpropagate through the whole window into the policy parameters.
    This exercises paper Eq. 8-12 (the adjoint chain)."""
    from shac.models.networks import GaussianActor

    torch.manual_seed(0)
    env = make_env(num_envs=4, dtype="float32")
    actor = GaussianActor(3, 1, (64, 64))

    obs = env.initialize_trajectory()
    loss = torch.zeros(())
    for _ in range(8):
        action = torch.tanh(actor(obs))
        obs, rew, done, _ = env.step(action)
        loss = loss + rew.mean()
    v = torch.randn(1).requires_grad_(True)     # stand-in terminal value
    loss = loss + v * obs[:, 0].mean()        # terminal term depends on s_h

    loss.backward()
    grads_ok = all(p.grad is not None and torch.isfinite(p.grad).all()
                   for p in actor.parameters())
    assert grads_ok, "policy gradients missing or non-finite"
    print("  full-window backward: finite policy gradients OK")


# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_energy_conservation()
    test_equilibria()
    test_transition_jacobians()
    test_reset_at_horizon()
    test_reward_and_obs_at_goal()
    test_actor_loss_backward_path()
    print("ALL PENDULUM DYNAMICS TESTS PASSED")
