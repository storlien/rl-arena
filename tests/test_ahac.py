"""
Tests for the AHAC-INSPIRED extension (NOT native SHAC) --
shac/algorithms/shac.py's adaptive-horizon dual, the iterative-critic
early-stop criterion, the env stiffness API, and the double-critic
construction.

1. DUAL DYNAMICS (ahac_dual_update, the exact official transcription):
   contact-free envs (cfs = 0 < C) drive lambda negative without bound and
   ratchet H down to steps_min, where it STAYS (the mechanism's domain
   boundary on this repo's two tasks -- AHAC.md Prediction 0); stiff contact
   (cfs > C) pushes lambda positive and H up; H clips at both ends; lambda
   is re-tiled to the new window length each step.
2. ITERATIVE-CRITIC CRITERION (ahac_critic_converged): flatlining windows
   stop, still-moving windows don't, short windows never stop.
3. STIFFNESS API: contact-free pendulum = exact zeros; cartpole = 0 well
   inside the travel limits, > 0 beyond them, and consistent with the
   limit force the dynamics actually apply.
4. DOUBLE-CRITIC CONSTRUCTION (--double-critic): two critics, NO target
   network, min pooling -- the AHAC delta 2 == SAPO choice IV equivalence.
5. ADAPTIVE SMOKE RUN: a few episodes of full-AHAC training on cartpole
   complete with finite losses; H stays at steps_min with cfs ~ 0 (the
   logged Prediction-0 evidence); the CSV schema carries the horizon
   columns (None in fixed mode); save/load restores H and lambda.

Run directly:  uv run python tests/test_ahac.py
"""

import csv
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shac.config import EnvConfig, PendulumEnvConfig, SHACConfig, RunConfig
from shac.algorithms.shac import (SHACTrainer, ahac_dual_update,
                                  ahac_critic_converged)
from shac.envs import make_env
from shac.utils.logger import RunLogger


def test_dual_contact_free_ratchet():
    print("test 1: contact-free dual -- lambda drifts down, H ratchets to "
          "steps_min and stays")
    torch.manual_seed(0)
    C, lr = 500.0, 5e-4
    lambd = torch.zeros(16)
    H = 16.0
    for _ in range(200):
        # contract: cfs_mean tracks the CURRENT window length (the trainer
        # slices cfs_buf[:self.h] -- same length as lambda)
        cfs = torch.zeros(lambd.shape[0])     # contact-free: cfs = 0 < C
        lambd, H = ahac_dual_update(lambd, H, cfs, C, lr, 8, 64)
        assert H >= 8.0 and H <= 64.0
    assert H == 8.0, f"H should pin at steps_min, got {H}"
    # once pinned, it can never recover: lambda keeps decreasing (unclamped)
    l_before = lambd.clone()
    lambd, H = ahac_dual_update(lambd, H, cfs, C, lr, 8, 64)
    assert H == 8.0 and bool((lambd < l_before).all())
    # the drift rate: every step loses exactly lr * C per dual variable
    # (and lambda is re-tiled to round(H) = 8 copies of lambda[0])
    lambd2 = torch.zeros(4)
    lambd2, H2 = ahac_dual_update(lambd2, 8.0, torch.zeros(4), C, lr, 8, 64)
    assert lambd2.shape[0] == 8 and H2 == 8.0
    assert torch.allclose(lambd2, torch.full((8,), -lr * C))
    print("  ok")


def test_dual_stiff_contact_grows_H():
    print("test 2: stiff contact (cfs > C) -- lambda grows, H grows")
    C, lr = 500.0, 5e-4
    lambd = torch.zeros(16)
    H = 16.0
    for _ in range(200):
        cfs = torch.full((lambd.shape[0],), C + 10000.0)  # gross violation
        lambd, H = ahac_dual_update(lambd, H, cfs, C, lr, 8, 64)
    assert H == 64.0, f"H should pin at steps_max, got {H}"
    assert float(lambd.mean()) > 0.0
    print("  ok")


def test_dual_lambda_retiling():
    print("test 3: lambda is re-tiled to the new window length (official "
          "init_buffers)")
    C, lr = 500.0, 5e-4
    lambd = torch.full((16,), -0.25)          # lambda[0] = -0.25
    H = 16.0
    # mild contact above threshold -> H grows, lambda tiles to the new length
    cfs = torch.full((lambd.shape[0],), C + 1.0)
    lambd, H = ahac_dual_update(lambd, H, cfs, C, lr, 8, 64)
    assert lambd.shape[0] == round(H)
    assert torch.allclose(lambd, torch.full((round(H),), -0.25 + lr * 1.0))
    print("  ok")


def test_critic_convergence_criterion():
    print("test 4: iterative-critic early-stop criterion")
    tol = 0.2
    # flatlined: |mean of the 4 diffs| = 0.0475 < 0.2 -> converged
    flat = [1.0, 0.9, 0.85, 0.82, 0.81]
    assert ahac_critic_converged(flat, tol)
    # still descending: |mean diff| = 0.2375 >= 0.2 -> not converged
    moving = [1.0, 0.5, 0.2, 0.1, 0.05]
    assert not ahac_critic_converged(moving, tol)
    # fewer than 5 pass losses -> never converged yet
    assert not ahac_critic_converged([1.0, 0.99, 0.98], tol)
    assert not ahac_critic_converged([], tol)
    # sign-agnostic: rising-but-flat also stops (official uses abs())
    rising_flat = [1.0, 1.01, 1.02, 1.03, 1.031]
    assert ahac_critic_converged(rising_flat, tol)
    print("  ok")


def test_stiffness_api():
    print("test 5: env stiffness API -- pendulum zeros, cartpole limit-gated")
    dev = torch.device("cpu")
    # pendulum: contact-free by construction
    pend = make_env("pendulum",
                    PendulumEnvConfig(num_envs=8, episode_length=10), dev)
    assert torch.equal(pend.stiffness(torch.zeros(8, 1)),
                       torch.zeros(8))
    # cartpole: zero while the cart is inside its limits, positive beyond
    cp = make_env("cartpole", EnvConfig(num_envs=4, episode_length=10), dev)
    cp.q[:, 0] = 0.0                     # mid-track: no limit engagement
    s = cp.stiffness(torch.zeros(4, 1))
    assert torch.equal(s, torch.zeros(4))
    cp.q[:, 0] = torch.tensor([4.5, 3.9, -4.5, 0.0])   # one env over +limit
    s = cp.stiffness(torch.zeros(4, 1))
    assert float(s[0]) > 0.0 and float(s[1]) == 0.0
    assert float(s[2]) > 0.0 and float(s[3]) == 0.0
    # consistency: stiffness ~ |f_limit| / max(|x_ddot|, 1) on the cart DoF
    force = cp._limit_force(cp.q, cp.qd)
    acc = cp._accelerations(cp.q, cp.qd, torch.zeros(4))
    expect = (force.abs() / acc[:, 0].abs().clamp(min=1.0))
    assert torch.allclose(s, expect, atol=1e-6)
    print("  ok")


def test_double_critic_construction():
    print("test 6: --double-critic == ensemble 2 + no target + min pool")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(critic_ensemble=2, use_target_critic=False,
                          actor_bootstrap_pool="min")
    torch.manual_seed(0)
    t = SHACTrainer(env_cfg, algo_cfg, RunConfig(seed=0, device="cpu",
                                                 logdir=None, eval_interval=0,
                                                 save_interval=0),
                    torch.device("cpu"))
    assert len(t.critics) == 2 and t.target_critics == []
    # min-pooled bootstrap: pooled value == elementwise min of the two critics
    obs = torch.randn(5, t.num_obs)
    with torch.no_grad():
        v1, v2 = t.critics[0](obs).squeeze(-1), t.critics[1](obs).squeeze(-1)
        pooled = t._pool_bootstrap(obs, "min")
    assert torch.allclose(pooled, torch.minimum(v1, v2))
    # and no Polyak step happens (stage 4 is a no-op without targets)
    t._update_target_critic()               # must not raise / must no-op
    print("  ok")


def test_adaptive_smoke_run():
    print("test 7: full-AHAC smoke run -- H ratchets to steps_min, cfs ~ 0, "
          "schema + save/load")
    with tempfile.TemporaryDirectory() as tmp:
        env_cfg = EnvConfig(num_envs=8, episode_length=16)
        algo_cfg = SHACConfig(horizon_mode="adaptive", steps_min=8,
                              steps_max=64, num_episodes=3,
                              critic_ensemble=2, use_target_critic=False,
                              iterative_critic=True,
                              iterative_critic_max=3)
        run_cfg = RunConfig(seed=0, device="cpu", logdir=tmp,
                            eval_interval=0, save_interval=0)
        logger = RunLogger(tmp, meta={}, append=False)
        t = SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"),
                        env_name="cartpole", logger=logger)
        t.train()
        # buffers sized for the capacity, window at the minimum
        assert t.obs_buf.shape[0] == 64 and t.h == 8
        # contact-free cartpole: H pinned at steps_min from episode 1
        with open(os.path.join(tmp, "log.csv")) as f:
            rows = list(csv.DictReader(f))
        assert len(rows) == 3
        assert all(float(r["horizon_H"]) == 8.0 for r in rows)
        assert all(abs(float(r["cfs_mean"])) < 1e-6 for r in rows)
        assert all(float(r["lambd_mean"]) < 0.0 for r in rows)
        # finite losses throughout
        assert all(float(r["actor_loss"]) == float(r["actor_loss"])
                   and abs(float(r["actor_loss"])) < 1e6 for r in rows)
        # fixed-horizon schema parity: same columns, None cells
        with tempfile.TemporaryDirectory() as tmp2:
            algo2 = SHACConfig(num_episodes=1)
            logger2 = RunLogger(tmp2, meta={}, append=False)
            t2 = SHACTrainer(env_cfg, algo2, RunConfig(seed=0, device="cpu",
                                                       logdir=tmp2,
                                                       eval_interval=0,
                                                       save_interval=0),
                             torch.device("cpu"), env_name="cartpole",
                             logger=logger2)
            t2.train()
            with open(os.path.join(tmp2, "log.csv")) as f:
                rows2 = list(csv.DictReader(f))
            assert set(rows[0].keys()) == set(rows2[0].keys())
            assert all(r2["horizon_H"] == "" for r2 in rows2)
        # save/load restores H and lambda into a fresh adaptive trainer
        t.save(suffix="probe")
        algo3 = SHACConfig(horizon_mode="adaptive", steps_min=8, steps_max=64,
                           num_episodes=3, critic_ensemble=2,
                           use_target_critic=False, iterative_critic=True,
                           iterative_critic_max=3)
        t3 = SHACTrainer(env_cfg, algo3, run_cfg, torch.device("cpu"),
                         env_name="cartpole", logger=logger)
        t3.load(os.path.join(tmp, "probe.pt"))
        assert t3.h == t.h and t3.horizon_float == t.horizon_float
        assert torch.allclose(t3.lambd, t.lambd)
    print("  ok")


if __name__ == "__main__":
    test_dual_contact_free_ratchet()
    test_dual_stiff_contact_grows_H()
    test_dual_lambda_retiling()
    test_critic_convergence_criterion()
    test_stiffness_api()
    test_double_critic_construction()
    test_adaptive_smoke_run()
    print("\nall AHAC tests passed")
