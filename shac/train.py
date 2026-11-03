"""
Command-line training entry point.

Examples
--------
    # CPU (plenty for CartPole at N=64)
    uv run python -m shac.train --device cpu

    # GPU (any CUDA device)
    uv run python -m shac.train --device cuda:0

    # Short smoke run
    uv run python -m shac.train --device cpu --episodes 50

Outputs are written to  runs/cartpole_<timestamp>/
    log.csv           one row per learning episode (plot with scripts/)
    metadata.json     the full resolved configuration
    best_policy.pt    checkpoint with the best evaluation return
    final_policy.pt   checkpoint at the end of training
"""

import argparse
import datetime
import json
from dataclasses import asdict

import torch

from shac.config import EnvConfig, SHACConfig, RunConfig
from shac.algorithms.shac import SHACTrainer
from shac.utils.logger import RunLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAC on CartPole Swing-Up")

    # ---- where / how to run -------------------------------------------
    p.add_argument("--device", default="auto",
                   help="'auto', 'cpu', 'cuda', 'cuda:0', ... (default: auto)")
    p.add_argument("--threads", type=int, default=1,
                   help="torch CPU intra-op threads. The tiny tensor ops in "
                        "this workload are FASTER with few threads (thread "
                        "pool contention dominates on many-core machines). "
                        "Ignored on GPU.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logdir", default=None,
                   help="output directory (default: runs/cartpole_<timestamp>)")

    # ---- the important algorithm knobs (defaults = paper Table 4) -------
    p.add_argument("--episodes", type=int, default=500,
                   help="M: number of learning episodes (paper: 500)")
    p.add_argument("--num-envs", type=int, default=64,
                   help="N: parallel environments (paper: 64)")
    p.add_argument("--horizon", type=int, default=32,
                   help="h: short-horizon window length (paper: 32)")
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--td-lambda", type=float, default=0.95)
    p.add_argument("--actor-lr", type=float, default=1e-2)
    p.add_argument("--critic-lr", type=float, default=1e-3)
    p.add_argument("--target-alpha", type=float, default=0.2,
                   help="target critic Polyak alpha (paper CartPole: 0.2)")

    # ---- evaluation / logging cadence ------------------------------------
    p.add_argument("--eval-interval", type=int, default=10,
                   help="evaluate every k episodes (0 disables)")
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--save-interval", type=int, default=100)
    return p.parse_args()


def resolve_device(name: str) -> torch.device:
    """Turn the --device argument into a torch.device.

    'auto' picks CUDA when available, otherwise CPU. Everything in this
    implementation (environments included) is written with plain tensor
    ops, so the same code runs unchanged on either device."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("CUDA not available -- falling back to CPU "
              "(fine for CartPole).")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.threads))

    # ---- assemble configuration from CLI overrides -----------------------
    env_cfg = EnvConfig(num_envs=args.num_envs)
    algo_cfg = SHACConfig(
        short_horizon=args.horizon,
        num_episodes=args.episodes,
        gamma=args.gamma,
        td_lambda=args.td_lambda,
        actor_learning_rate=args.actor_lr,
        critic_learning_rate=args.critic_lr,
        target_critic_alpha=args.target_alpha,
    )
    run_cfg = RunConfig(
        seed=args.seed,
        device=str(device),
        logdir=args.logdir,
        eval_interval=args.eval_interval,
        eval_num_episodes=args.eval_episodes,
        save_interval=args.save_interval,
    )

    # ---- reproducibility ---------------------------------------------------
    torch.manual_seed(run_cfg.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(run_cfg.seed)

    logdir = run_cfg.logdir or (
        "runs/cartpole_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    logger = RunLogger(logdir, meta={
        "env": asdict(env_cfg),
        "algo": asdict(algo_cfg),
        "run": asdict(run_cfg),
        "device": str(device),
    })

    print(f"SHAC -- CartPole Swing-Up")
    print(f"  device      : {device}")
    print(f"  N x h       : {env_cfg.num_envs} x {algo_cfg.short_horizon} "
          f"({env_cfg.num_envs * algo_cfg.short_horizon} env-steps/episode)")
    print(f"  episodes M  : {algo_cfg.num_episodes}")
    print(f"  logdir      : {logdir}")
    print(f"  task        : H = {env_cfg.episode_length} steps, "
          f"reward = -theta^2 - 0.1*thd^2 - 0.05*x^2 - 0.1*xd^2")
    print()

    trainer = SHACTrainer(env_cfg, algo_cfg, run_cfg, device, logger=logger)
    try:
        result = trainer.train()
    except KeyboardInterrupt:
        print("\nInterrupted -- saving current policy.")
        trainer.save(suffix="final_policy")
        return
    print(f"\nTraining finished. Best eval return: "
          f"{result['best_eval_return']:.1f}")
    print(f"Artifacts in: {logdir}")


if __name__ == "__main__":
    main()
