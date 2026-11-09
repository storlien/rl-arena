"""
Behavioral metrics for trained checkpoints (CartPole Swing-Up, Pendulum
Swing-Up, ... -- dispatched from the checkpoint's env_name).

Goes beyond the scalar return and measures WHAT the policy does: does it
swing up, how long does the swing-up take, and how well does it balance
once upright. Used for the no-critic ablation write-up (ABLATION.md), where
two policies can have similar returns while behaving very differently.

Usage:
    uv run python scripts/policy_metrics.py runs/ablation/a_shac_s0/best_policy.pt
    uv run python scripts/policy_metrics.py runs/ablation/*/best_policy.pt
    uv run python scripts/policy_metrics.py <ckpts...> --json

For every checkpoint it runs `--episodes` (default 64) DETERMINISTIC full
task episodes (policy mean, tanh-squashed) on a fresh environment and
reports:

    return_mean +/- std     undiscounted episode return (sum of r over H=240)
    swingup_success_rate     fraction of episodes in which the pole reaches
                            |theta| < 0.25 rad (14 deg) and HOLDS it there
                            for >= 10 consecutive steps
    median_time_to_upright   median over SUCCESSFUL episodes of the first
                            step of that 10-step upright streak (steps;
                            n/a if no episode succeeded)
    balance_offset_deg       mean over SUCCESSFUL episodes of the mean
                            |theta| over the final 60 steps (how upright it
                            ends up holding; n/a if none succeeded)

Headless-safe (pure tensor math, no rendering).
"""

import argparse
import glob
import json
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from shac.envs import make_env
from shac.utils.checkpoints import build_actor_and_rms, read_checkpoint

UPRIGHT_RAD = 0.25     # within 14 deg of upright counts as "up"
HOLD_STEPS = 10        # ... and it must hold for this many consecutive steps
BALANCE_WINDOW = 60    # mean |theta| over the last N steps -> balance offset

# Where (sin, cos) of the wrapped angle live in each env's observation.
#   cartpole: [x, x_dot, sin, cos, theta_dot]
#   pendulum: [cos, sin, theta_dot]
SIN_COS_IDX = {
    "cartpole": (2, 3),
    "pendulum": (1, 0),
}


def load_policy(path, device):
    """Load a checkpoint -> (env_name, env_cfg, ckpt). The actor and the
    obs normalizer are rebuilt by evaluate_checkpoint once the env exists
    (shac.utils.checkpoints)."""
    return read_checkpoint(path, device)


@torch.no_grad()
def evaluate_checkpoint(path, device, num_episodes, seed):
    """Run `num_episodes` deterministic episodes; return a metrics dict."""
    torch.manual_seed(seed)
    env_name, env_cfg, ckpt = load_policy(path, device)
    env = make_env(env_name, env_cfg, device, num_envs=num_episodes)
    actor, rms = build_actor_and_rms(ckpt, env, device)

    sin_i, cos_i = SIN_COS_IDX[env_name]
    obs = env.reset()
    returns = torch.zeros(num_episodes, device=device)
    thetas = []                      # |theta| per step, [T, num_episodes]
    # Exactly H steps per episode: correct for the current envs (no early
    # termination), unlike SHACTrainer.evaluate which also handles tasks
    # that can end early. If such an env is added, revisit this loop.
    for _ in range(env.episode_length):
        o = rms.normalize(obs) if rms is not None else obs
        action = torch.tanh(actor(o, deterministic=True))
        obs, rew, done, _ = env.step(action)
        returns += rew
        theta = torch.atan2(obs[:, sin_i], obs[:, cos_i]).abs()
        thetas.append(theta)

    abs_theta = torch.stack(thetas)                    # [T, E]
    T, E = abs_theta.shape
    up = abs_theta < UPRIGHT_RAD                       # [T, E]

    # first step of a >=HOLD_STEPS consecutive upright streak, per episode
    # (all state tensors on `env.device` -- a CPU/device mix here would
    # crash on CUDA while working fine on CPU)
    hold = torch.zeros(E, dtype=torch.long, device=device)
    first_upright = torch.full((E,), -1, dtype=torch.long, device=device)
    for t in range(T):
        hold = torch.where(up[t], hold + 1, torch.zeros_like(hold))
        newly = (hold >= HOLD_STEPS) & (first_upright < 0)
        first_upright = torch.where(newly, torch.full_like(first_upright, t
                                                           - HOLD_STEPS + 1),
                                    first_upright)
    success = first_upright >= 0

    times = first_upright[success]
    med_time = (float(times.float().median()) if success.any() else None)
    if success.any():
        balance = abs_theta[T - BALANCE_WINDOW:, :][:, success].mean(dim=0)
        balance_deg = float(balance.mean()) * 180.0 / math.pi
    else:
        balance_deg = None

    ret = returns.tolist()
    n = len(ret)
    mean = sum(ret) / n
    std = (sum((r - mean) ** 2 for r in ret) / n) ** 0.5
    return {
        "checkpoint": path,
        "episodes": num_episodes,
        "return_mean": mean,
        "return_std": std,
        "swingup_success_rate": float(success.float().mean()),
        "median_time_to_upright": med_time,
        "balance_offset_deg": balance_deg,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("checkpoints", nargs="+",
                   help="one or more .pt checkpoints (globs allowed)")
    p.add_argument("--episodes", type=int, default=64,
                   help="deterministic episodes per checkpoint (default 64)")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the episodes' initial states")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", action="store_true",
                   help="print one JSON object per checkpoint instead of a "
                        "table (for aggregation)")
    args = p.parse_args()

    paths = []
    for pattern in args.checkpoints:
        expanded = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") \
            else [pattern]
        paths.extend(expanded)
    if not paths:
        sys.exit("no checkpoints matched")

    device = torch.device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(1)

    metrics = [evaluate_checkpoint(pt, device, args.episodes, args.seed)
               for pt in paths]

    if args.json:
        for m in metrics:
            print(json.dumps(m))
        return

    hdr = (f"{'checkpoint':52s} {'return':>16s} {'swingup':>8s} "
           f"{'t_up':>6s} {'balance':>9s}")
    print(hdr)
    print("-" * len(hdr))
    for m in metrics:
        t_up = (f"{m['median_time_to_upright']:.0f}"
                if m["median_time_to_upright"] is not None else "n/a")
        bal = (f"{m['balance_offset_deg']:.1f} deg"
               if m["balance_offset_deg"] is not None else "n/a")
        print(f"{os.path.relpath(m['checkpoint']):52s} "
              f"{m['return_mean']:8.1f}+/-{m['return_std']:<6.1f} "
              f"{m['swingup_success_rate'] * 100:6.1f}% "
              f"{t_up:>6s} {bal:>9s}")


if __name__ == "__main__":
    main()
