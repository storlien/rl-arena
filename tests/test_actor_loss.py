"""
Correctness test for the actor loss (paper Eq. 5) accumulated inside
SHACTrainer._rollout_and_update_actor.

The trainer accumulates the segmented discounted window return with running
per-env tensors (rew_acc, gamma_pow) plus the terminal bootstrap
gamma * gamma_pow * V'(s_{t+1}), vectorized over envs. This test recomputes
the SAME quantity with a naive per-env scalar loop -- no broadcasting, no
fancy indexing -- and requires agreement on

  (1) a done-free window (defaults: H=240, h=32), and
  (2) a window with four synchronized task resets (H=8, h=32), including a
      done on the window's final step.

History: this test was written after a broadcasting bug was found in the
per-env accumulator reset,

    rew_acc = torch.where(done.unsqueeze(-1), 0, rew_acc)   # -> [N, N] !!

which silently turned the [N] accumulator into an [N, N] matrix: every env's
rewards got weighted by EVERY env's discount power, inflating the actor
loss by exactly N (=64) in done-free windows and scrambling the per-env
discount resets in windows containing dones. All runs trained before the
fix were regenerated.

Run directly:  uv run python tests/test_actor_loss.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from shac.config import EnvConfig, SHACConfig, RunConfig
from shac.algorithms.shac import SHACTrainer


def naive_eq5(rew_buf, nv_buf, done_mask, gamma):
    """Naive per-env reference for the trainer's actor loss (Eq. 5).

    Walks each env separately with plain Python floats -- structurally
    incapable of the broadcasting a vectorized version could get wrong.
    Returns the un-normalized (negative) sum, i.e. exactly what the trainer
    computes before its 1/(N*h) division:
    for each env:  -sum over segments of [ discounted segment return
                                       + gamma * gamma_pow * V'(terminal) ].
    """
    h, n = rew_buf.shape
    loss = 0.0
    for j in range(n):
        acc, gp = 0.0, 1.0
        for i in range(h):
            acc += gp * float(rew_buf[i, j])
            if i < h - 1 and float(done_mask[i, j]) > 0:
                # env finished its task episode mid-window: contribute its
                # segment return + terminal value, then start a fresh segment
                loss -= acc + gamma * gp * float(nv_buf[i, j])
                acc, gp = 0.0, 1.0
                continue
            if i == h - 1:
                # window end: every env contributes its remaining segment
                loss -= acc + gamma * gp * float(nv_buf[i, j])
            gp *= gamma
    return loss


def run_one_window(env_cfg, algo_cfg):
    torch.manual_seed(0)
    run_cfg = RunConfig(seed=0, device="cpu", logdir=None,
                        eval_interval=0, eval_num_episodes=4, save_interval=0)
    trainer = SHACTrainer(env_cfg, algo_cfg, run_cfg, torch.device("cpu"))
    stats = trainer._rollout_and_update_actor()
    return trainer, stats


def check(env_cfg, algo_cfg, expect_dones):
    h = algo_cfg.short_horizon
    n = env_cfg.num_envs
    gamma = algo_cfg.gamma
    trainer, stats = run_one_window(env_cfg, algo_cfg)

    has_dones = bool((trainer.done_mask[:-1] > 0).any())
    assert has_dones == expect_dones, (
        f"expected dones-in-window={expect_dones}, got {has_dones}")

    expected = naive_eq5(trainer.rew_buf, trainer.next_value_buf,
                         trainer.done_mask, gamma) / (n * h)
    got = stats["actor_loss"]
    assert abs(got - expected) <= 1e-3 * max(1.0, abs(expected)), (
        f"actor_loss mismatch: trainer {got:.6f} vs naive {expected:.6f} "
        f"(ratio {got / expected:.3f})")

    assert torch.isfinite(torch.tensor(got)), "actor loss is not finite"
    print(f"  window contains dones: {has_dones}"
          f"  trainer {got:12.6f}   naive {expected:12.6f}   OK")


def main():
    print("test 1: done-free window (H=240, h=32, N=64)")
    check(EnvConfig(), SHACConfig(num_episodes=10), expect_dones=False)

    print("test 2: four synchronized resets inside one window (H=8, h=32)")
    check(EnvConfig(episode_length=8), SHACConfig(num_episodes=10),
          expect_dones=True)

    print("\nALL ACTOR-LOSS TESTS PASSED")


if __name__ == "__main__":
    main()
