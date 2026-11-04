"""
Tests for the --no-critic ablation (SHACConfig.use_critic=False).

1. LOSS EQUIVALENCE -- the ablation must equal "SHAC with the bootstrap
   removed": a full-SHAC trainer whose TARGET critic is identically zero
   produces nv == 0 everywhere, so its actor loss and actor gradients must
   match the no-critic trainer's exactly (same actor weights, same reseeded
   rollout). Any divergence means the no-critic path is not implementing
   "Eq. 5 without the V' term". The window deliberately contains task
   resets (H=16 < h=32) so the mid-window contribution path is exercised
   in both trainers.

2. SMOKE -- a few episodes of no-critic training complete with finite
   losses, critic_loss is logged as an empty CSV cell (the schema is
   identical to full SHAC), checkpoints store None in the critic keys,
   and load() restores actor + obs_rms into a fresh no-critic trainer.

Run directly:  uv run python tests/test_no_critic.py
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


def build(use_critic, env_cfg, algo_cfg):
    """Build a trainer deterministically (same RNG -> same actor init)."""
    torch.manual_seed(123)
    run_cfg = RunConfig(seed=0, device="cpu", logdir=None,
                        eval_interval=0, eval_num_episodes=4, save_interval=0)
    return SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"))


def test_loss_equivalence():
    print("test 1: zeroed-target-critic SHAC == no-critic trainer")
    env_cfg = EnvConfig(num_envs=16, episode_length=16)
    algo_kwargs = dict(short_horizon=32, num_episodes=10)

    t_full = build(True, env_cfg, SHACConfig(use_critic=True, **algo_kwargs))
    with torch.no_grad():                       # zero the TARGET critic ->
        for p in t_full.target_critic.parameters():   # nv == 0 everywhere
            p.zero_()
    t_abl = build(False, env_cfg, SHACConfig(use_critic=False, **algo_kwargs))

    # same actor weights by construction
    sd1, sd2 = t_full.actor.state_dict(), t_abl.actor.state_dict()
    assert sd1.keys() == sd2.keys()
    for k in sd1:
        assert torch.equal(sd1[k], sd2[k]), f"actor weights differ: {k}"
    # no critic objects at all in the ablated trainer
    assert t_abl.critic is None and t_abl.target_critic is None
    assert t_abl.critic_opt is None

    torch.manual_seed(777)                      # identical rollouts
    s_full = t_full._rollout_and_update_actor()
    g_full = [p.grad.clone() for p in t_full.actor.parameters()]
    torch.manual_seed(777)
    s_abl = t_abl._rollout_and_update_actor()
    g_abl = [p.grad.clone() for p in t_abl.actor.parameters()]

    assert s_full["actor_loss"] == s_abl["actor_loss"], (
        f"actor losses differ: {s_full['actor_loss']} vs {s_abl['actor_loss']}")
    for a, b in zip(g_full, g_abl):
        assert torch.allclose(a, b, rtol=0.0, atol=1e-7), (
            "actor gradients differ between zeroed-critic SHAC and ablation")
    assert math.isfinite(s_abl["actor_loss"])
    print(f"  actor_loss (identical): {s_abl['actor_loss']:.6f}")
    print("  actor gradients: identical")


def test_smoke():
    print("test 2: no-critic smoke run (3 episodes, checkpoint round-trip)")
    logdir = tempfile.mkdtemp(prefix="test_no_critic_", dir="runs")
    env_cfg = EnvConfig(num_envs=8, episode_length=16)
    algo_cfg = SHACConfig(num_episodes=3, use_critic=False)
    run_cfg = RunConfig(seed=0, device="cpu", logdir=logdir,
                        eval_interval=2, eval_num_episodes=4, save_interval=0)
    logger = RunLogger(logdir, meta={"test": "no-critic smoke"})
    trainer = SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"),
                         logger=logger)
    result = trainer.train()
    assert "best_eval_return" in result

    # CSV: same schema as full SHAC, but critic_loss cells are empty
    with open(os.path.join(logdir, "log.csv")) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3
    assert all(r["critic_loss"] == "" for r in rows), (
        "critic_loss should be an empty cell in every no-critic row")
    assert all(math.isfinite(float(r["actor_loss"])) for r in rows)
    assert all(math.isfinite(float(r["actor_grad_norm"])) for r in rows)
    assert rows[1]["eval_return_mean"] != "", "eval at episode 2 missing"
    print(f"  3 episodes trained; eval return at ep 2: "
          f"{float(rows[1]['eval_return_mean']):.1f}")

    # checkpoint: critic keys stored as None
    ckpt = torch.load(os.path.join(logdir, "final_policy.pt"),
                      map_location="cpu", weights_only=False)
    assert ckpt["critic"] is None and ckpt["target_critic"] is None
    assert ckpt["critic_opt"] is None
    assert ckpt["actor"] is not None

    # load() round-trip into a fresh no-critic trainer
    t2 = SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"))
    t2.load(os.path.join(logdir, "final_policy.pt"))
    for k in ckpt["actor"]:
        assert torch.equal(t2.actor.state_dict()[k], ckpt["actor"][k])
    assert t2.obs_rms is not None and t2.obs_rms.count == ckpt["obs_rms"]["count"]
    print("  checkpoint stores critic=None; actor + obs_rms round-trip OK")


def main():
    test_loss_equivalence()
    test_smoke()
    print("\nALL NO-CRITIC TESTS PASSED")


if __name__ == "__main__":
    main()
