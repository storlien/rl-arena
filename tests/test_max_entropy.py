"""
Tests for the maximum-entropy objective (SHACConfig.max_entropy):
SAC's entropy-augmented return + dual-tuned alpha, instantiated the way
SAPO does it (entropy added to the actor's window return AND to the
critic's td-(lambda) targets, H-bar-normalized inside the losses, RAW
entropy in the dual).

1. ENTROPY ESTIMATOR -- tanh_gaussian_entropy is checked against a
   Monte-Carlo estimate of the TRUE squashed-Gaussian entropy
   (-log pi via the change of variables). The estimator drops only the
   zero-mean E[eps^2] term, so its mean must match the MC truth; it must
   stay BELOW the pre-tanh Gaussian entropy (squashing concentrates),
   approach it for tiny sigma (tanh inactive), and be differentiable in
   both mu and sigma.

2. DUAL STEP -- SAC's Eq. 21 direction: at init the policy entropy is
   ABOVE the target H-bar, so alpha must DECREASE; forced tiny sigma
   puts it BELOW the target, so alpha must INCREASE; the step clamps
   at alpha >= 0. The logged entropy_alpha is the PRE-dual alpha that
   entered the actor loss.

3. ACTOR-LOSS WIRING -- same-seed plain and entropy trainers roll out
   BIT-IDENTICAL windows (sample_with_entropy shares forward()'s RNG
   path), and the returned losses differ by EXACTLY the accumulated
   entropy contribution _window_ent_contrib / (N*h). The entropy term
   also adds a DIRECT gradient onto the sigma parameter (the
   counter-pressure whose absence collapsed sigma(s) in the state-std
   A/B).

4. CRITIC WIRING -- _update_critic fits SOFT values: its td-(lambda)
   targets are computed from r + alpha * hat{h} with the POST-dual
   alpha, and they differ from the plain targets.

5. TRAIN INTEGRATION -- a full tiny train() run: the CSV schema is
   IDENTICAL with and without the flag (entropy_alpha / policy_entropy
   columns always present, empty cells when disabled), alpha starts at
   the init value and decreases, and checkpoints round-trip the
   dual-tuned alpha (restored by max-entropy trainers, ignored by
   plain ones, absent/None in plain checkpoints).

Run directly:  uv run python tests/test_max_entropy.py
"""

import csv
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shac.config import EnvConfig, SHACConfig, RunConfig
from shac.models.networks import (GaussianActor, tanh_gaussian_entropy,
                                  TANH_ENTROPY_EPS)
import shac.algorithms.shac as shac_mod
from shac.algorithms.shac import SHACTrainer, td_lambda_values
from shac.utils.logger import RunLogger

GAUSS_ENT_CONST = 0.5 * math.log(2.0 * math.pi * math.e)   # 1.4189...


def build(env_cfg, algo_cfg, run_cfg, logger=None):
    """Build a trainer deterministically (same RNG -> same init)."""
    torch.manual_seed(123)
    return SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"),
                       logger=logger)


def tiny_cfg(**kw):
    kw.setdefault("short_horizon", 8)
    kw.setdefault("num_episodes", 5)
    return SHACConfig(**kw)


def test_entropy_estimator():
    print("test 1: tanh_gaussian_entropy vs Monte-Carlo ground truth")
    torch.manual_seed(0)
    n = 200_000
    sigma = 0.7
    eps = torch.randn(n, 1)
    u = sigma * eps                                   # mu = 0
    a = torch.tanh(u)
    # TRUE per-sample -log pi(a): change of variables
    #   log pi(a) = log N(u; mu, sigma) - log(1 - a^2)
    log_n = (-0.5 * eps ** 2 - math.log(sigma)
             - 0.5 * math.log(2.0 * math.pi))
    mc = float((-log_n + torch.log(1 - a ** 2 + TANH_ENTROPY_EPS)).mean())
    est = float(tanh_gaussian_entropy(
        u, torch.full((n, 1), sigma)).mean())
    # the estimator replaces the E[eps^2]=1 term by its mean: unbiased,
    # so the two MC means agree up to sampling noise (~1e-3 here)
    assert abs(est - mc) < 0.01, (
        f"estimator mean {est:.4f} != MC true entropy {mc:.4f} "
        f"(bias {(est - mc):.4f} -- sign of the tanh correction?)")
    # squashing CONCENTRATES: entropy stays below the pre-tanh Gaussian's
    gauss = math.log(sigma) + GAUSS_ENT_CONST
    assert est < gauss, (f"squashed entropy {est:.4f} must be below the "
                         f"Gaussian's {gauss:.4f}")

    # tiny sigma: tanh is inactive, entropy -> the Gaussian's
    sigma = 0.01
    u = sigma * torch.randn(n, 1)
    est = float(tanh_gaussian_entropy(
        u, torch.full((n, 1), sigma)).mean())
    assert abs(est - (math.log(sigma) + GAUSS_ENT_CONST)) < 0.01, (
        f"tiny-sigma entropy should be ~Gaussian, got {est:.4f}")

    # large sigma: actions saturate, the tanh correction dominates and
    # the entropy collapses far below the Gaussian's
    sigma = 2.0
    u = sigma * torch.randn(n, 1)
    est = float(tanh_gaussian_entropy(
        u, torch.full((n, 1), sigma)).mean())
    assert est < gauss_limit(sigma) - 1.0, (
        f"saturation should cut entropy far below the Gaussian's "
        f"(got {est:.3f} vs {gauss_limit(sigma):.3f})")

    # differentiable in mu (pushes away from saturation) and in sigma
    mu = torch.zeros(16, 1, requires_grad=True)
    log_sig = torch.zeros(16, 1, requires_grad=True)
    u = mu + log_sig.exp() * torch.randn(16, 1)
    h = tanh_gaussian_entropy(u, log_sig.exp()).sum()
    h.backward()
    assert mu.grad is not None and float(mu.grad.abs().max()) > 0.0, (
        "no gradient into mu: the tanh correction is severed")
    assert log_sig.grad is not None and float(log_sig.grad.abs().max()) > 0.0, (
        "no gradient into sigma: the log sigma term is severed")
    print(f"  sigma 0.7: estimator mean == MC truth ({mc:+.4f}); "
          f"tiny sigma -> Gaussian entropy; sigma 2.0 -> collapsed far "
          f"below it; grads reach mu and sigma")


def gauss_limit(sigma):
    """Entropy of the PRE-tanh Gaussian N(0, sigma^2)."""
    return math.log(sigma) + GAUSS_ENT_CONST


def test_dual_step():
    print("test 2: the alpha dual moves the right way (SAC Eq. 21)")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    t = build(env_cfg, tiny_cfg(max_entropy=True),
              RunConfig(seed=0, device="cpu", eval_interval=0,
                        save_interval=0))
    hbar = t.entropy_target
    assert hbar == -0.5, f"CartPole default target is -0.5, got {hbar}"

    # ---- at init: sigma = exp(-1) -> h ~ +0.3 ABOVE the target -> the
    #      dual must LOWER alpha from its init value ----
    stats = t._rollout_and_update_actor()
    assert stats["entropy_alpha"] == 1.0, (
        "the first episode's actor loss must use the INIT alpha (pre-dual)")
    assert stats["policy_entropy"] is not None
    assert t.entropy_alpha < 1.0, (
        f"h > H-bar at init, dual must decrease alpha, got {t.entropy_alpha}")
    assert stats["policy_entropy"] > hbar, (
        "init entropy should be above the target for this direction test")
    alpha_after_init = t.entropy_alpha

    # ---- forced tiny sigma: h << H-bar -> the dual must RAISE alpha ----
    with torch.no_grad():
        t.actor.logstd.fill_(-6.0)                 # sigma = e^-6 ~ 0.0025
    stats = t._rollout_and_update_actor()
    assert stats["policy_entropy"] < hbar, (
        "tiny sigma should push the entropy below the target")
    assert t.entropy_alpha > stats["entropy_alpha"], (
        f"h < H-bar must INCREASE alpha: {stats['entropy_alpha']} -> "
        f"{t.entropy_alpha}")

    # ---- clamp: a huge dual step cannot push alpha below zero ----
    t2 = build(env_cfg, tiny_cfg(max_entropy=True, entropy_lr=100.0),
               RunConfig(seed=0, device="cpu", eval_interval=0,
                         save_interval=0))
    t2._rollout_and_update_actor()
    assert t2.entropy_alpha == 0.0, (
        f"alpha must clamp at >= 0, got {t2.entropy_alpha}")
    print(f"  init: alpha 1.000 -> {alpha_after_init:.4f} (h above target); "
          f"tiny sigma: {stats['entropy_alpha']:.4f} -> "
          f"{t.entropy_alpha:.4f} (h below target, raised); "
          f"clamp: alpha == {t2.entropy_alpha:.1f}")


def test_actor_loss_wiring():
    print("test 3: same-seed windows are bit-identical; the loss gap is "
          "exactly the entropy contribution")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    run_cfg = RunConfig(seed=0, device="cpu", eval_interval=0,
                        save_interval=0)
    t_plain = build(env_cfg, tiny_cfg(), run_cfg)
    t_ent = build(env_cfg, tiny_cfg(max_entropy=True), run_cfg)
    # both builds seed(123) and construct identically -> they leave the
    # GLOBAL torch RNG in the same state; replay that state before each
    # window so the two rollouts sample the SAME noise stream (this is
    # the actual claim: sample_with_entropy consumes the RNG exactly
    # like forward -- one randn per step, nothing else).
    rng_state = torch.get_rng_state()

    torch.set_rng_state(rng_state)
    loss_plain_t = t_plain._run_window()
    torch.set_rng_state(rng_state)
    loss_ent_t = t_ent._run_window()
    N, h = t_ent.num_envs, t_ent.h
    loss_plain, loss_ent = (loss_plain_t.detach().item(),
                            loss_ent_t.detach().item())

    # the trajectories themselves are bit-identical: same observations,
    # same rewards, same terminal values (only the loss carries the
    # extra entropy part)
    assert torch.equal(t_plain.obs_buf, t_ent.obs_buf), (
        "observations diverged: sample_with_entropy consumes RNG "
        "differently than forward")
    assert torch.equal(t_plain.rew_buf, t_ent.rew_buf), (
        "reward streams diverged")
    assert torch.equal(t_plain.next_value_buf, t_ent.next_value_buf)

    # => the two losses differ by EXACTLY the accumulated entropy part
    contrib = t_ent._window_ent_contrib / (N * h)
    assert abs((loss_plain - loss_ent) - contrib) < 1e-5, (
        f"loss gap {loss_plain - loss_ent:.6f} != entropy contribution "
        f"{contrib:.6f} (rollouts diverged or the term is miswired)")
    assert abs(contrib) > 1e-3, "entropy contribution should be nonzero"

    # the entropy term adds a DIRECT gradient onto the sigma parameter
    # (in addition to the eps path) -- the counter-pressure that was
    # missing in the state-std A/B
    t_plain.actor_opt.zero_grad()
    t_ent.actor_opt.zero_grad()
    loss_plain_t.backward()
    loss_ent_t.backward()
    g_plain = t_plain.actor.logstd.grad
    g_ent = t_ent.actor.logstd.grad
    assert g_plain is not None and g_ent is not None
    assert not torch.allclose(g_plain, g_ent), (
        "the entropy term must add gradient on sigma beyond the "
        "reparameterization path")
    print(f"  loss plain {loss_plain:.4f} vs entropy {loss_ent:.4f}: "
          f"gap == alpha*hat{chr(72)} contribution {contrib:.6f} exactly; "
          f"sigma gradients differ (entropy term reaches logstd)")


def test_critic_wiring():
    print("test 4: critic targets are SOFT (r + alpha*hat{h}, post-dual)")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    t = build(env_cfg, tiny_cfg(max_entropy=True),
              RunConfig(seed=0, device="cpu", eval_interval=0,
                        save_interval=0))
    stats = t._rollout_and_update_actor()       # fills buffers, dual fires
    alpha_post = t.entropy_alpha
    assert alpha_post != stats["entropy_alpha"], (
        "the critic must see the POST-dual alpha (dual already ran)")

    # spy on the td-(lambda) call inside _update_critic
    captured = {}
    orig = shac_mod.td_lambda_values

    def spy(rew_buf, *args, **kw):
        captured["rew"] = rew_buf
        out = orig(rew_buf, *args, **kw)
        captured["targets"] = out
        return out
    shac_mod.td_lambda_values = spy
    try:
        t._update_critic()
    finally:
        shac_mod.td_lambda_values = orig

    expected = t.rew_buf + alpha_post * t.ent_buf
    assert torch.equal(captured["rew"], expected), (
        "critic targets must run on r + alpha*hat{h} (post-dual alpha)")
    assert not torch.equal(captured["rew"], t.rew_buf), (
        "the soft term is missing: critic saw the plain rewards")
    plain = orig(t.rew_buf, t.next_value_buf, t.done_mask,
                 t.cfg.gamma, t.cfg.td_lambda)
    diff = float((captured["targets"] - plain).abs().max())
    assert diff > 0.0, "soft and plain targets are identical?"
    print(f"  captured soft reward == rew + {alpha_post:.4f}*ent exactly; "
          f"targets differ from plain by up to {diff:.3f}")


def test_train_integration():
    print("test 5: train() integration -- CSV schema, alpha schedule, "
          "checkpoint round-trip")
    # ---- max-entropy arm ----
    logdir = tempfile.mkdtemp(prefix="test_max_ent_", dir="runs")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = tiny_cfg(max_entropy=True, num_episodes=4)
    run_cfg = RunConfig(seed=0, device="cpu", logdir=logdir,
                        eval_interval=0, save_interval=0,
                        grad_var_interval=0)
    t = build(env_cfg, algo_cfg, run_cfg,
              RunLogger(logdir, meta={"test": "max-entropy integration"}))
    t.train()

    with open(os.path.join(logdir, "log.csv")) as f:
        reader = csv.DictReader(f)
        header_ent = reader.fieldnames
        rows = list(reader)
    assert "entropy_alpha" in header_ent and "policy_entropy" in header_ent
    # alpha used by episode 1's actor loss is the init value; the dual
    # then moves it DOWN (h > H-bar at init)
    assert float(rows[0]["entropy_alpha"]) == 1.0
    assert float(rows[1]["entropy_alpha"]) < 1.0
    assert all(r["policy_entropy"] != "" for r in rows), "H column dropped"
    h0 = float(rows[0]["policy_entropy"])
    assert -0.5 < h0 < 1.5, f"init entropy ~0.2 expected, got {h0}"

    # ---- checkpoint round-trip of the dual-tuned alpha ----
    path = t.save("final_policy")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    assert ckpt["entropy_alpha"] is not None
    assert ckpt["entropy_alpha"] == t.entropy_alpha
    t2 = build(env_cfg, algo_cfg, run_cfg)
    t2.entropy_alpha = 999.0                     # sentinel: must be restored
    t2.load(path)
    assert t2.entropy_alpha == t.entropy_alpha, (
        "load() did not restore the dual-tuned alpha")
    # a NON-entropy trainer loads the same checkpoint and ignores alpha
    t3 = build(env_cfg, tiny_cfg(num_episodes=4), run_cfg)
    t3.load(path)
    assert t3.entropy_alpha == float(t3.cfg.entropy_init_alpha), (
        "plain trainer must ignore the checkpoint's alpha")

    # ---- plain arm: identical schema, empty entropy cells ----
    logdir_p = tempfile.mkdtemp(prefix="test_plain_ent_", dir="runs")
    algo_p = tiny_cfg(num_episodes=4)
    run_p = RunConfig(seed=0, device="cpu", logdir=logdir_p,
                      eval_interval=0, save_interval=0, grad_var_interval=0)
    tp = build(env_cfg, algo_p, run_p,
               RunLogger(logdir_p, meta={"test": "plain"}))
    tp.train()
    with open(os.path.join(logdir_p, "log.csv")) as f:
        reader = csv.DictReader(f)
        header_plain = reader.fieldnames
        rows_p = list(reader)
    assert header_plain == header_ent, (
        "CSV schema must be identical with and without --max-entropy "
        f"\nplain:  {header_plain}\nentropy: {header_ent}")
    assert all(r["entropy_alpha"] == "" and r["policy_entropy"] == ""
               for r in rows_p), "disabled columns must be empty cells"
    ckpt_p = torch.load(tp.save("final_policy"), map_location="cpu",
                        weights_only=False)
    assert ckpt_p["entropy_alpha"] is None, (
        "plain checkpoints must store None alpha")

    # ---- --state-std + --max-entropy coexist (the SAPO arm) ----
    logdir_s = tempfile.mkdtemp(prefix="test_sapo_", dir="runs")
    algo_s = tiny_cfg(max_entropy=True, actor_state_std=True,
                      num_episodes=2)
    run_s = RunConfig(seed=0, device="cpu", logdir=logdir_s,
                      eval_interval=0, save_interval=0, grad_var_interval=0)
    ts = build(env_cfg, algo_s, run_s,
               RunLogger(logdir_s, meta={"test": "sapo"}))
    stats = ts._rollout_and_update_actor()
    assert 0.0 < stats["policy_std"] < 1.0 and stats["entropy_alpha"] == 1.0
    print(f"  schema identical across arms; alpha 1.0 -> "
          f"{float(rows[1]['entropy_alpha']):.4f}; H_1 = {h0:+.3f}; "
          f"ckpt alpha round-trips; state-std + max-entropy coexist")


def main():
    test_entropy_estimator()
    test_dual_step()
    test_actor_loss_wiring()
    test_critic_wiring()
    test_train_integration()
    print("\nALL MAX-ENTROPY TESTS PASSED")


if __name__ == "__main__":
    main()
