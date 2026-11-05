"""
Tests for the SAC-style STATE-DEPENDENT policy std
(SHACConfig.actor_state_std / GaussianActor(state_std=True)).

1. INIT EQUIVALENCE -- the state-std actor must start from EXACTLY
   sigma(s) = exp(actor_logstd_init) at every state (the log-std half of
   the head is zero-initialized with bias logstd_init), so the A/B
   against the global-std variant is clean. deterministic=True returns
   mu(s) only, and the architectures differ as documented (no `logstd`
   parameter in state mode).

2. STATE DEPENDENCE -- after perturbing the log-std half of the head,
   sigma varies across states, is a deterministic function of the obs
   (same obs -> same sigma), and the SAC clamp bounds hold
   (sigma in [exp(-20), exp(2)]).

3. GRADIENT PATH -- one real training window must put gradient into the
   log-std half of the head (dL/dsigma(s) = dL/du * eps of the
   reparameterized path) and Adam must move it: the SHAC loss has no
   entropy/log-prob term, so this is the ONLY way sigma(s) can learn.

4. TRAIN INTEGRATION -- a full tiny train() run logs policy_std as the
   window-mean of sigma(s): the first row is exactly exp(logstd_init),
   later rows move, and a deterministic evaluate() never feeds the
   accumulator (std() reads 0.0 after it). The same tiny run in the
   DEFAULT (global) mode still trains and starts at the same sigma --
   the flag changes only the sigma parameterization.

5. CHECKPOINT ROUND-TRIP -- save/load preserves mu(s) and sigma(s)
   exactly, the algo_cfg metadata carries actor_state_std (what
   scripts/render_policy.py keys off), and loading a state-std
   checkpoint into a global-std actor fails on the shape mismatch,
   as documented.

Run directly:  uv run python tests/test_state_std.py
"""

import math
import os
import sys
import tempfile

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shac.config import EnvConfig, SHACConfig, RunConfig
from shac.models.networks import GaussianActor, LOG_STD_MIN, LOG_STD_MAX
from shac.algorithms.shac import SHACTrainer
from shac.utils.logger import RunLogger

LOGSTD_INIT = -1.0
SIGMA0 = math.exp(LOGSTD_INIT)


def head(actor):
    """The bare Linear output layer of the actor MLP."""
    return [m for m in actor.mu_net.modules() if isinstance(m, nn.Linear)][-1]


def obs_batch(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    # raw CartPole-like states; the actor only needs consistent tensors
    return torch.randn(n, 5, generator=g) * torch.tensor([1., 1., 1., 1., 1.])


def build(env_cfg, algo_cfg, run_cfg, logger=None):
    """Build a trainer deterministically (same RNG -> same actor init)."""
    torch.manual_seed(123)
    return SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"),
                      logger=logger)


def test_init_equivalence():
    print("test 1: state-std actor starts at sigma(s) == exp(logstd_init)")
    actor = GaussianActor(5, 1, (64, 64), LOGSTD_INIT, state_std=True)
    actor.eval()

    obs = obs_batch(32)
    with torch.no_grad():
        mu, sigma = actor.mu_sigma(obs)
        det = actor(obs, deterministic=True)

    # every state starts from the SAME sigma = exp(logstd_init)
    assert torch.allclose(sigma, torch.full_like(sigma, SIGMA0)), (
        f"sigma(s) at init should be exactly exp({LOGSTD_INIT}) = {SIGMA0}, "
        f"got min {sigma.min():.6f} max {sigma.max():.6f}")
    # deterministic path returns mu only
    assert torch.allclose(det, mu), "deterministic forward must return mu(s)"
    # the zero-init logstd half: log_std(s) == logstd_init exactly
    h = head(actor)
    assert torch.all(h.weight[1:].abs() == 0), (
        "log-std half of the head must start zero-initialized")
    assert torch.all(h.bias[1:] == LOGSTD_INIT)
    # architecture differs from the global mode as documented
    assert not hasattr(actor, "logstd"), (
        "state-std actor must not carry the global logstd parameter")
    assert h.out_features == 2, "head must emit [mu | log_std(s)]"

    # global mode: same start, but sigma is a parameter, not a function
    g = GaussianActor(5, 1, (64, 64), LOGSTD_INIT)
    with torch.no_grad():
        _, gs = g.mu_sigma(obs)
    assert torch.allclose(gs, torch.full_like(gs, SIGMA0))
    assert hasattr(g, "logstd") and g.logstd.shape == (1,)
    print(f"  sigma(s) = {SIGMA0:.6f} at all 32 states; deterministic == mu; "
          f"no global logstd parameter")


def test_state_dependence():
    print("test 2: sigma becomes state-dependent after perturbing the head")
    actor = GaussianActor(5, 1, (64, 64), LOGSTD_INIT, state_std=True)
    actor.eval()
    h = head(actor)
    with torch.no_grad():
        h.weight[1:].normal_(0.0, 0.5)      # give log_std a state pathway
        h.bias[1:].normal_(0.0, 0.3)

    obs = obs_batch(64)
    with torch.no_grad():
        _, sigma = actor.mu_sigma(obs)
        _, sigma_again = actor.mu_sigma(obs[:8])

    spread = float(sigma.max() - sigma.min())
    assert spread > 1e-3, (
        f"sigma should vary across states after perturbation, "
        f"spread = {spread:.2e}")
    # sigma is a deterministic function of the state
    assert torch.allclose(sigma_again, sigma[:8]), (
        "same obs must give the same sigma(s)")
    # SAC clamp bounds: sigma in [exp(LOG_STD_MIN), exp(LOG_STD_MAX)]
    # (relative tolerance: torch.exp is float32, math.exp is float64)
    assert float(sigma.min()) >= math.exp(LOG_STD_MIN) * (1 - 1e-6)
    assert float(sigma.max()) <= math.exp(LOG_STD_MAX) * (1 + 1e-6)

    # clamping is active at the bounds, not just by chance
    with torch.no_grad():
        h.weight[1:].zero_()
        h.bias[1:] = 50.0                    # far above LOG_STD_MAX = 2
        assert torch.allclose(actor.mu_sigma(obs)[1],
                              torch.full((64, 1), math.exp(LOG_STD_MAX)))
        h.bias[1:] = -50.0                   # far below LOG_STD_MIN = -20
        assert torch.allclose(actor.mu_sigma(obs)[1],
                              torch.full((64, 1), math.exp(LOG_STD_MIN)))
    print(f"  sigma varies across states (spread {spread:.4f}), "
          f"is a deterministic function of s, and respects "
          f"[exp({LOG_STD_MIN}), exp({LOG_STD_MAX})]")


def test_gradient_path():
    print("test 3: the window gradient reaches the log-std half of the head")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(short_horizon=8, num_episodes=5,
                          actor_state_std=True)
    run_cfg = RunConfig(seed=0, device="cpu", eval_interval=0,
                        save_interval=0)
    t = build(env_cfg, algo_cfg, run_cfg)

    h = head(t.actor)
    bias_before = h.bias.detach().clone()
    row_before = h.weight[1:].detach().clone()

    t._rollout_and_update_actor()            # one real training episode

    # gradient reached the log-std half via the reparameterized noise
    assert h.weight.grad is not None
    g_logstd = h.weight.grad[1:]
    assert float(g_logstd.abs().max()) > 0.0, (
        "no gradient in the log-std half: sigma(s) could never learn "
        "(the reparameterization eps-term is the only gradient path -- "
        "SHAC has no entropy/log-prob term)")
    # and Adam actually moved it
    dw = (h.weight[1:].detach() - row_before).abs().max()
    db = (h.bias[1:].detach() - bias_before).abs().max()
    assert float(dw) > 0.0 and float(db) > 0.0, (
        f"Adam did not move the log-std half (dw={dw:.2e}, db={db:.2e})")
    # policy_std drained the window accumulator (8 stochastic forwards)
    stats = t._rollout_and_update_actor()
    assert stats["policy_std"] > 0.0
    print(f"  |grad| in log-std half: {float(g_logstd.abs().max()):.3e}; "
          f"Adam moved it (max dw={float(dw):.2e}); "
          f"policy_std={stats['policy_std']:.6f}")


def test_train_integration():
    print("test 4: train() logs the window-mean sigma(s); eval stays clean")
    logdir = tempfile.mkdtemp(prefix="test_state_std_", dir="runs")
    logger = RunLogger(logdir, meta={"test": "state-std integration"})
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(short_horizon=8, num_episodes=4,
                          actor_state_std=True)
    run_cfg = RunConfig(seed=0, device="cpu", logdir=logdir,
                        eval_interval=0, save_interval=0,
                        grad_var_interval=2, grad_var_samples=2)
    t = build(env_cfg, algo_cfg, run_cfg, logger=logger)
    t.train()

    with open(os.path.join(logdir, "log.csv")) as f:
        import csv
        rows = list(csv.DictReader(f))
    stds = [float(r["policy_std"]) for r in rows]

    assert len(stds) == 4 and all(s > 0 for s in stds)
    # episode 1 rolls out with the initial weights: exactly exp(logstd_init)
    # (tolerance = float32 exp rounding, ~1 ulp)
    assert abs(stds[0] - SIGMA0) < 1e-6, (
        f"first policy_std should be exactly exp({LOGSTD_INIT}) = {SIGMA0}, "
        f"got {stds[0]}")
    # the head's log-std half learned: sigma moved off the init value
    assert abs(stds[-1] - stds[0]) > 1e-6, (
        f"policy_std never moved: {stds}")

    # deterministic evaluation must not feed the accumulator
    t.actor.reset_std()
    t.evaluate(2)
    assert t.actor.std() == 0.0, (
        "evaluate() (deterministic) polluted the policy_std accumulator")

    # ---- default (global) mode regression: same tiny run, same start ----
    logdir_g = tempfile.mkdtemp(prefix="test_global_std_", dir="runs")
    algo_g = SHACConfig(short_horizon=8, num_episodes=4)
    run_g = RunConfig(seed=0, device="cpu", logdir=logdir_g,
                      eval_interval=0, save_interval=0)
    tg = build(env_cfg, algo_g, run_g, RunLogger(logdir_g,
                                                 meta={"test": "global"}))
    tg.train()
    with open(os.path.join(logdir_g, "log.csv")) as f:
        import csv
        stds_g = [float(r["policy_std"]) for r in csv.DictReader(f)]
    assert abs(stds_g[0] - SIGMA0) < 5e-3, (
        f"global mode should start at the same sigma, got {stds_g[0]}")
    assert hasattr(tg.actor, "logstd")
    print(f"  policy_std: {[f'{s:.4f}' for s in stds]} "
          f"(first == exp(-1) exactly, then moves); "
          f"global mode starts at {stds_g[0]:.4f}")


def test_checkpoint_roundtrip():
    print("test 5: checkpoint round-trip preserves mu(s) and sigma(s)")
    logdir = tempfile.mkdtemp(prefix="test_state_std_ckpt_", dir="runs")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(short_horizon=8, num_episodes=3,
                          actor_state_std=True)
    run_cfg = RunConfig(seed=0, device="cpu", logdir=logdir,
                        eval_interval=0, save_interval=0)
    t = build(env_cfg, algo_cfg, run_cfg, RunLogger(logdir,
                                                    meta={"test": "ckpt"}))
    t.train()
    path = t.save("final_policy")

    # the algo_cfg metadata carries the flag (what the scripts key off)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["algo_cfg"]["actor_state_std"] is True

    # fresh trainer (same config) loads and reproduces mu(s), sigma(s)
    t2 = build(env_cfg, algo_cfg, run_cfg)
    t2.load(path)
    obs = obs_batch(32)
    t.actor.eval(); t2.actor.eval()
    with torch.no_grad():
        mu1, s1 = t.actor.mu_sigma(obs)
        mu2, s2 = t2.actor.mu_sigma(obs)
    assert torch.equal(mu1, mu2) and torch.equal(s1, s2), (
        "loaded policy differs from the saved one")

    # the state-std checkpoint cannot load into a GLOBAL-std actor
    # (documented architecture incompatibility)
    g = GaussianActor(5, 1, (64, 64), LOGSTD_INIT, state_std=False)
    try:
        g.load_state_dict(ckpt["actor"])
        raise AssertionError("state-std checkpoint loaded into a "
                             "global-std actor (shape check missing?)")
    except RuntimeError:
        pass

    # the scripts' loading pattern reconstructs the right architecture
    g2 = GaussianActor(5, 1, tuple(ckpt["algo_cfg"]["actor_units"]),
                       ckpt["algo_cfg"]["actor_logstd_init"],
                       state_std=ckpt["algo_cfg"].get("actor_state_std",
                                                     False))
    g2.load_state_dict(ckpt["actor"])
    print("  mu(s), sigma(s) identical after round-trip; "
          "actor_state_std in metadata; cross-mode load correctly fails")


def main():
    test_init_equivalence()
    test_state_dependence()
    test_gradient_path()
    test_train_integration()
    test_checkpoint_roundtrip()
    print("\nALL STATE-STD TESTS PASSED")


if __name__ == "__main__":
    main()
