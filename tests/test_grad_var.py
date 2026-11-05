"""
Tests for the gradient-estimator variance diagnostic
(RunConfig.grad_var_interval / grad_var_samples,
 SHACTrainer._measure_grad_variance).

1. PURITY -- sampling the estimator must NOT change anything training
   depends on: actor weights, optimizer state, existing .grad buffers and
   total_env_steps all have to be bit-identical before/after. (The k
   windows deliberately DO advance the environment stream and the
   observation normalizer -- that is shared bookkeeping with real
   windows, not pollution.)

2. SAME-ESTIMATOR -- a k=1 measurement with the same seed must reproduce
   the training update's gradient EXACTLY: the diagnostic samples the very
   gradient the trainer would backprop, not some other loss.

3. CSV SCHEMA -- with grad_var_interval=2 the log gets sparse
   actor_grad_var / actor_grad_mean_norm columns (values every 2nd
   episode, empty between, like the eval columns), a dense
   critic_grad_norm column, and env_steps that does NOT count the extra
   diagnostic windows (episode * N * h exactly). A --no-critic run keeps
   the identical schema with empty critic cells.

Run directly:  uv run python tests/test_grad_var.py
"""

import csv
import math
import os
import sys
import tempfile

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shac.config import EnvConfig, SHACConfig, RunConfig
from shac.algorithms.shac import SHACTrainer
from shac.utils.logger import RunLogger


def build(env_cfg, algo_cfg, run_cfg):
    """Build a trainer deterministically (same RNG -> same actor init)."""
    torch.manual_seed(123)
    return SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"))


def test_purity():
    print("test 1: measurement leaves training state bit-identical")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(short_horizon=8, num_episodes=5)
    run_cfg = RunConfig(seed=0, device="cpu", eval_interval=0, save_interval=0)
    t = build(env_cfg, algo_cfg, run_cfg)

    t._rollout_and_update_actor()          # train one episode: Adam state
    weights = [p.detach().clone() for p in t.actor.parameters()]
    grads = [p.grad.clone() for p in t.actor.parameters()]
    opt_state = t.actor_opt.state_dict()
    steps_before = t.total_env_steps

    out = t._measure_grad_variance(4)

    for w, p in zip(weights, t.actor.parameters()):
        assert torch.equal(w, p.detach()), "measurement changed actor weights"
    for g, p in zip(grads, t.actor.parameters()):
        assert torch.equal(g, p.grad), (
            "measurement touched the .grad buffers of the real update "
            "(autograd.grad must not)")
    assert t.actor_opt.state_dict() == opt_state, (
        "measurement changed optimizer state")
    assert t.total_env_steps == steps_before, (
        "diagnostic windows must not count as training env steps")

    assert math.isfinite(out["actor_grad_var"])
    assert math.isfinite(out["actor_grad_mean_norm"])
    assert out["actor_grad_var"] >= 0.0
    assert out["actor_grad_var"] > 0.0, (
        "k=4 windows of a stochastic policy must have positive "
        "estimator variance (exploration noise is resampled)")
    # NOTE: v-hat is bounded by E[||g||^2] (the mean SQUARED sample norm),
    # NOT by ||E[g]||^2 -- v-hat > ||E[g]||^2 simply means one window's
    # gradient is noise-dominated, which is the expected regime for a
    # fresh random policy (and the paper's motivation for SHAC).
    print(f"  weights/grads/optimizer/env_steps untouched; "
          f"var={out['actor_grad_var']:.4g}, "
          f"||E[g]||={out['actor_grad_mean_norm']:.4g}")


def test_same_estimator():
    print("test 2: k=1 measurement == the training update's gradient")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(short_horizon=8, num_episodes=5)
    run_cfg = RunConfig(seed=0, device="cpu", eval_interval=0, save_interval=0)

    torch.manual_seed(777)
    t_train = build(env_cfg, algo_cfg, run_cfg)
    # stats["actor_grad_norm"] is the PRE-clip norm (what clip_grad_norm_
    # returns); p.grad itself has already been rescaled to norm
    # grad_norm_clip by the clip, so the pre-clip column is the right
    # thing to compare the (pre-clip) sampled gradient against.
    s = t_train._rollout_and_update_actor()
    norm_train = s["actor_grad_norm"]

    torch.manual_seed(777)                 # identical weights + RNG stream
    t_meas = build(env_cfg, algo_cfg, run_cfg)
    out = t_meas._measure_grad_variance(1)

    assert abs(out["actor_grad_mean_norm"] - norm_train) < 1e-6 * norm_train, (
        f"k=1 sampled gradient differs from the training gradient: "
        f"{out['actor_grad_mean_norm']} vs {norm_train}")
    assert out["actor_grad_var"] == 0.0    # one sample -> zero scatter
    print(f"  ||g|| matches the training draw: {norm_train:.6f}")


def test_csv_schema():
    print("test 3: CSV schema, sparsity and env_steps accounting")
    logdir = tempfile.mkdtemp(prefix="test_grad_var_", dir="runs")
    logger = RunLogger(logdir, meta={"test": "grad-var schema"})

    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(short_horizon=8, num_episodes=5)
    run_cfg = RunConfig(seed=0, device="cpu", logdir=logdir,
                        eval_interval=0, save_interval=0,
                        grad_var_interval=2, grad_var_samples=3)
    t = SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"),
                    logger=logger)
    t.train()

    with open(os.path.join(logdir, "log.csv")) as f:
        reader = csv.DictReader(f)
        header = reader.fieldnames
        rows = list(reader)

    for name in ("actor_grad_var", "actor_grad_mean_norm",
                 "critic_grad_norm"):
        assert name in header, f"{name} missing from the CSV header"
    # sparse like the eval columns: full on every 2nd episode, empty between
    for i, r in enumerate(rows, start=1):
        full = (r["actor_grad_var"] != ""
                and r["actor_grad_mean_norm"] != "")
        assert full == (i % 2 == 0), f"sparsity wrong on row {i}"
        assert r["critic_grad_norm"] != "", f"critic_grad_norm empty on {i}"
        assert math.isfinite(float(r["critic_grad_norm"]))
        # diagnostic windows are NOT training samples
        assert int(r["env_steps"]) == i * env_cfg.num_envs * algo_cfg.short_horizon

    # --no-critic arm: identical schema, empty critic cells
    logdir2 = tempfile.mkdtemp(prefix="test_grad_var_nc_", dir="runs")
    logger2 = RunLogger(logdir2, meta={"test": "grad-var no-critic"})
    algo_nc = SHACConfig(short_horizon=8, num_episodes=4, use_critic=False)
    run_nc = RunConfig(seed=0, device="cpu", logdir=logdir2,
                       eval_interval=0, save_interval=0,
                       grad_var_interval=2, grad_var_samples=3)
    t2 = SHACTrainer(env_cfg, algo_nc, run_nc, torch.device("cpu"),
                     logger=logger2)
    t2.train()
    with open(os.path.join(logdir2, "log.csv")) as f:
        reader = csv.DictReader(f)
        header2 = reader.fieldnames
        rows2 = list(reader)
    assert header2 == header, "no-critic run must keep the identical schema"
    assert all(r["critic_grad_norm"] == "" and r["critic_loss"] == ""
               for r in rows2)
    assert rows2[1]["actor_grad_var"] != "", "variance missing in ablation"
    print(f"  schema consistent across arms; variance logged every "
          f"{run_cfg.grad_var_interval} episodes")


def main():
    test_purity()
    test_same_estimator()
    test_csv_schema()
    print("\nALL GRAD-VAR TESTS PASSED")


if __name__ == "__main__":
    main()
