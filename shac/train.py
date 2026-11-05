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
    p.add_argument("--no-critic", action="store_true",
                   help="ablation: train the actor WITHOUT a critic -- the "
                        "loss keeps only the pathwise-derivative "
                        "short-horizon window reward of paper Eq. 5 (no "
                        "terminal bootstrap, no td-(lambda), no target "
                        "network); i.e. discounted short-horizon BPTT")
    p.add_argument("--state-std", action="store_true",
                   help="experimental A/B: SAC-style STATE-DEPENDENT policy "
                        "std (the paper's sigma_theta(s) notation) -- the "
                        "actor head emits [mu | log_std(s)], clamped to "
                        "[-20, 2] as in SAC, instead of the official global "
                        "learnable sigma. Zero-init head: both variants start "
                        "from the same sigma = exp(logstd_init)")
    p.add_argument("--max-entropy", action="store_true",
                   help="experimental arm: SAC/SAPO-style MAXIMUM-ENTROPY "
                        "objective -- augment the actor's window return and "
                        "the critic's td-(lambda) targets with "
                        "alpha * entropy (SAPO Eq. 18-20), alpha auto-tuned "
                        "by SAC's dual (Eq. 21) toward the target entropy. "
                        "Adds the entropy counter-pressure whose absence "
                        "collapsed sigma in the --state-std A/B "
                        "(STATE_STD.md). Works with BOTH sigma "
                        "parameterizations; logs entropy_alpha / "
                        "policy_entropy; checkpoints load across the flag")
    p.add_argument("--entropy-target", type=float, default=None,
                   help="target entropy H-bar for the alpha dual "
                        "(default: -dim(A)/2, SAC/SAPO convention; -0.5 for "
                        "CartPole's 1-D action)")
    p.add_argument("--entropy-lr", type=float, default=5e-3,
                   help="dual-step learning rate for alpha")
    p.add_argument("--entropy-init-alpha", type=float, default=1.0,
                   help="initial temperature alpha (SAPO Table 4)")

    # ---- evaluation / logging cadence ------------------------------------
    p.add_argument("--eval-interval", type=int, default=10,
                   help="evaluate every k episodes (0 disables)")
    p.add_argument("--eval-episodes", type=int, default=16)
    p.add_argument("--save-interval", type=int, default=100)

    # ---- gradient diagnostics ------------------------------------------
    p.add_argument("--grad-var-interval", type=int, default=25,
                   help="every k episodes, sample the actor gradient-"
                        "estimator with extra frozen-policy windows and "
                        "log its variance (actor_grad_var) and mean norm "
                        "(actor_grad_mean_norm); 0 disables")
    p.add_argument("--grad-var-samples", type=int, default=8,
                   help="extra windows per variance measurement (>= 2)")
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
        use_critic=not args.no_critic,
        actor_state_std=args.state_std,
        max_entropy=args.max_entropy,
        entropy_target=args.entropy_target,
        entropy_lr=args.entropy_lr,
        entropy_init_alpha=args.entropy_init_alpha,
    )
    run_cfg = RunConfig(
        seed=args.seed,
        device=str(device),
        logdir=args.logdir,
        eval_interval=args.eval_interval,
        eval_num_episodes=args.eval_episodes,
        save_interval=args.save_interval,
        grad_var_interval=args.grad_var_interval,
        grad_var_samples=args.grad_var_samples,
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
    print(f"  critic      : "
          f"{'ON (full SHAC)' if algo_cfg.use_critic else 'OFF -- ablation: '
           'pathwise derivative only'}")
    if algo_cfg.max_entropy:
        print(f"  max-entropy : ON -- soft window return + soft td-(lambda) "
              f"targets, dual-tuned alpha (target H = "
              f"{-0.5 if algo_cfg.entropy_target is None else algo_cfg.entropy_target:+.2f}"
              f" [-dim(A)/2 default, CartPole dim(A)=1], "
              f"init alpha = {algo_cfg.entropy_init_alpha:g})")
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
