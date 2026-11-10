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

Outputs are written to  runs/<env>_<timestamp>/
    log.csv                  one row per learning episode (plot with scripts/)
    metadata.json            the full resolved configuration
    best_policy.pt           checkpoint with the best evaluation return
    final_policy.pt          checkpoint at the end of training
    interrupted_policy.pt    checkpoint after Ctrl-C (resumable via --resume)
"""

import argparse
import datetime
import json
from dataclasses import asdict

import torch

from shac.config import EnvConfig, SHACConfig, RunConfig
from shac.algorithms.shac import SHACTrainer
from shac.envs import ENV_REGISTRY, make_env_config
from shac.utils.logger import RunLogger


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAC on differentiable envs")

    # ---- which task ----------------------------------------------------
    p.add_argument("--env", default="cartpole",
                   choices=sorted(ENV_REGISTRY.keys()),
                   help="which differentiable environment to train on "
                        "(default: cartpole, the paper's task; pendulum is "
                        "NOT in the paper -- see PENDULUM.md)")

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
                   help="output directory (default: runs/<env>_<timestamp>; "
                        "an existing logdir's log.csv is never overwritten)")
    p.add_argument("--resume", default=None, metavar="PATH",
                   help="resume training from a checkpoint saved by this "
                        "repo (weights, optimizer states, observation "
                        "statistics, dual-tuned alpha, episode counter and "
                        "best-eval tracker are restored; the RNG stream and "
                        "the simulator state are NOT, so a resumed run is "
                        "statistically but not bit-wise identical to an "
                        "uninterrupted one)")

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
    p.add_argument("--critic-warmup", type=int, default=0, metavar="W",
                   help="train the critic ONLY for the first W episodes "
                        "(TD3-style delayed policy update): the rollout "
                        "still runs but the actor gets no gradient step "
                        "and the alpha dual is skipped")
    p.add_argument("--critic-ensemble", type=int, default=1, metavar="K",
                   help="number of critics (SAPO design choice IV: K=2); "
                        "bootstrap values are min-pooled over the ensemble "
                        "(SAC's double-Q trick)")
    p.add_argument("--no-target-critic", action="store_true",
                   help="bootstrap from the ONLINE critics, no Polyak "
                        "target network (with --critic-ensemble 2 this is "
                        "exactly SAPO's choice IV)")
    p.add_argument("--activation", default="elu", choices=["elu", "silu"],
                   help="hidden-layer activation for actor and critic "
                        "(silu = SAPO design choice V)")
    p.add_argument("--optimizer", default="adam", choices=["adam", "adamw"],
                   help="optimizer for both networks (adamw = SAPO design "
                        "choice V; identical lr/betas, decoupled decay)")
    p.add_argument("--grad-clip", type=float, default=1.0, metavar="C",
                   help="actor gradient-norm clip (paper/official 1.0; "
                        "SAPO design choice V uses 0.5)")
    p.add_argument("--actor-bootstrap-pool", default="min",
                   choices=["min", "avg"],
                   help="how the critic ENSEMBLE is pooled in the actor's "
                        "window loss when --critic-ensemble > 1: 'min' "
                        "(pessimistic) or 'avg' (SAPO's paper: average in "
                        "the actor return, min only for critic targets)")
    # ---- AHAC-INSPIRED knobs (Georgiev et al., ICML 2024; all default to
    # ---- native SHAC behavior). Repo naming: single knobs = "SHAC +
    # ---- AHAC-inspired <component>"; --ahac (all three) = "AHAC".
    p.add_argument("--adaptive-horizon", action="store_true",
                   help="AHAC delta 1: adapt the window length H every "
                        "episode via the Lagrangian dual on contact "
                        "stiffness (paper Eq. 9-10). H starts at "
                        "--steps-min and is clipped to [--steps-min, "
                        "--steps-max]. WARNING (AHAC.md): on contact-free "
                        "tasks the dual ratchets H to steps_min")
    p.add_argument("--steps-min", type=int, default=8, metavar="HMIN",
                   help="AHAC: initial/minimum adaptive window (official 8)")
    p.add_argument("--steps-max", type=int, default=64, metavar="HMAX",
                   help="AHAC: maximum adaptive window (official 64)")
    p.add_argument("--contact-threshold", type=float, default=500.0,
                   metavar="C",
                   help="AHAC: stiffness threshold C (official 500)")
    p.add_argument("--lambd-lr", type=float, default=5e-4, metavar="LR",
                   help="AHAC: dual step size for (lambda, H); decays "
                        "linearly to 1e-5 like the network LRs (official)")
    p.add_argument("--double-critic", action="store_true",
                   help="AHAC delta 2: two independent online critics, "
                        "min-pooled bootstrap, NO Polyak target network. "
                        "Implemented as critic_ensemble=2 + "
                        "use_target_critic=False + pool=min (the same "
                        "mechanism as SAPO's design choice IV)")
    p.add_argument("--iterative-critic", action="store_true",
                   help="AHAC delta 3: train the critic to convergence "
                        "EVERY episode (up to 64 passes, early stop when "
                        "the last five pass losses flatline < 0.2) instead "
                        "of native SHAC's fixed 16 passes")
    p.add_argument("--ahac", action="store_true",
                   help="the FULL AHAC algorithm = --adaptive-horizon + "
                        "--double-critic + --iterative-critic (Georgiev et "
                        "al., ICML 2024). Use the individual flags for "
                        "ablations")
    p.add_argument("--init-mode", default="hanging",
                   choices=["hanging", "uniform", "upright"],
                   help="PENDULUM ONLY: initial-state distribution -- "
                        "'hanging' (repo convention, every episode a "
                        "genuine swing-up), 'uniform' (gym-faithful: "
                        "theta ~ U(-pi, pi), omega ~ U(-1, 1) -- a "
                        "built-in balance curriculum), or 'upright' "
                        "(near-upright balance-only curriculum)")
    p.add_argument("--energy-shaping", type=float, default=0.0, metavar="W",
                   help="PENDULUM ONLY: energy-shaping reward penalty "
                        "weight w in r -= w*((E - E*)/(2 m g l))^2 "
                        "(E* = m g l upright at rest); 0 = gym reward "
                        "exactly. Prices energy near the separatrix so "
                        "the orbit policy gets a gradient toward "
                        "arriving slow")
    p.add_argument("--actor-logstd-init", type=float, default=-1.0,
                   metavar="S",
                   help="initial log-std of the policy (default -1.0 -> "
                        "sigma0 = 0.37, paper). Higher = wider initial "
                        "exploration, e.g. 0.0 -> sigma0 = 1.0")
    p.add_argument("--ep-length", type=int, default=None, metavar="H",
                   help="override the TASK horizon H (episode_length; "
                        "default = the env's paper/gym value, 240 for both "
                        "tasks). PENDULUM: H=480 approximates gym's "
                        "10 s clock at this repo's dt=1/60")
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
        print("CUDA not available -- falling back to CPU.")
        return torch.device("cpu")
    return torch.device(name)


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)
    if device.type == "cpu":
        torch.set_num_threads(max(1, args.threads))

    # ---- assemble configuration from CLI overrides -----------------------
    env_cfg = make_env_config(args.env, num_envs=args.num_envs)
    if args.env == "pendulum":
        env_cfg.init_mode = args.init_mode
        env_cfg.energy_shaping = args.energy_shaping
    if args.ep_length is not None:
        env_cfg.episode_length = args.ep_length
    # ---- AHAC-INSPIRED composites ----------------------------------------
    # --double-critic (AHAC delta 2) needs no new trainer code: it is exactly
    # the SAPO-inspired choice-IV combination (two online critics, min-pooled
    # bootstrap, no Polyak target). Explicit flags win over the composite.
    if args.double_critic:
        args.critic_ensemble = 2
        args.no_target_critic = True
        args.actor_bootstrap_pool = "min"
    # --ahac = ALL THREE AHAC deltas together (the naming convention: this
    # -- and only this -- is "AHAC"; single knobs are "SHAC + AHAC-inspired
    # <component>").
    if args.ahac:
        args.adaptive_horizon = True
        args.iterative_critic = True
        args.double_critic = True
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
        actor_logstd_init=args.actor_logstd_init,
        max_entropy=args.max_entropy,
        entropy_target=args.entropy_target,
        entropy_lr=args.entropy_lr,
        entropy_init_alpha=args.entropy_init_alpha,
        critic_warmup=args.critic_warmup,
        critic_ensemble=args.critic_ensemble,
        use_target_critic=not args.no_target_critic,
        actor_bootstrap_pool=args.actor_bootstrap_pool,
        activation=args.activation,
        optimizer=args.optimizer,
        grad_norm_clip=args.grad_clip,
        # AHAC-INSPIRED switches (defaults = native SHAC behavior)
        horizon_mode=("adaptive" if args.adaptive_horizon else "fixed"),
        steps_min=args.steps_min,
        steps_max=args.steps_max,
        contact_threshold=args.contact_threshold,
        lambd_lr=args.lambd_lr,
        iterative_critic=args.iterative_critic,
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
        "runs/" + args.env + "_"
        + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"))
    # append=True when resuming: continue the run's log.csv (row schema
    # and header come from the original run) instead of refusing to touch it.
    logger = RunLogger(logdir, meta={
        "env_name": args.env,
        "env": asdict(env_cfg),
        "algo": asdict(algo_cfg),
        "run": asdict(run_cfg),
        "device": str(device),
    }, append=args.resume is not None)

    spec = ENV_REGISTRY[args.env]
    print(f"SHAC -- {spec.label}")
    print(f"  env         : {args.env}")
    print(f"  device      : {device}")
    print(f"  N x h       : {env_cfg.num_envs} x "
          f"{algo_cfg.steps_min if algo_cfg.horizon_mode == 'adaptive' else algo_cfg.short_horizon} "
          f"({env_cfg.num_envs * (algo_cfg.steps_min if algo_cfg.horizon_mode == 'adaptive' else algo_cfg.short_horizon)} env-steps/episode)")
    print(f"  episodes M  : {algo_cfg.num_episodes}")
    print(f"  logdir      : {logdir}")
    print(f"  task        : H = {env_cfg.episode_length} steps, "
          f"{spec.reward_desc}")
    print(f"  critic      : "
          f"{'ON (full SHAC)' if algo_cfg.use_critic else 'OFF -- ablation: '
           'pathwise derivative only'}")
    # AHAC-INSPIRED banner: which of the three deltas are active.
    ahac_deltas = [
        ("adaptive-horizon", algo_cfg.horizon_mode == "adaptive"),
        ("double-critic", args.double_critic),
        ("iterative-critic", algo_cfg.iterative_critic),
    ]
    if any(on for _, on in ahac_deltas):
        on = [name for name, flag in ahac_deltas if flag]
        label = "FULL AHAC" if len(on) == 3 else "SHAC + AHAC-inspired: "
        print(f"  ahac        : {label}{', '.join(on) if len(on) < 3 else ''}"
              f"{' [steps ' + str(algo_cfg.steps_min) + '..' + str(algo_cfg.steps_max) + ', C=' + str(algo_cfg.contact_threshold) + ', lam_lr=' + str(algo_cfg.lambd_lr) + ']' if algo_cfg.horizon_mode == 'adaptive' else ''}")
    if args.resume is not None:
        print(f"  resume      : {args.resume}")
    print()

    trainer = SHACTrainer(env_cfg, algo_cfg, run_cfg, device,
                          env_name=args.env, logger=logger)
    # The max-entropy banner line needs dim(A) for the default target; that
    # is an env property, so it is printed here (after the trainer exists)
    # rather than hardcoding "1 action dim" per env.
    if algo_cfg.max_entropy:
        target = (algo_cfg.entropy_target
                  if algo_cfg.entropy_target is not None
                  else -0.5 * trainer.num_actions)
        print(f"  max-entropy : ON -- soft window return + soft td-(lambda) "
              f"targets, dual-tuned alpha (target H = {target:+.2f} "
              f"[-dim(A)/2 default, dim(A) = {trainer.num_actions}], "
              f"init alpha = {algo_cfg.entropy_init_alpha:g})")
    try:
        result = trainer.train(resume=args.resume)
    except KeyboardInterrupt:
        # Distinct suffix: "final_policy" means the run COMPLETED its
        # episode budget; an interrupt is a different artifact.
        print("\nInterrupted -- saving current policy as "
              "interrupted_policy.pt (resumable with --resume).")
        trainer.save(suffix="interrupted_policy")
        return
    except FileExistsError as e:
        # The logger refuses to overwrite an existing run's log.csv; print
        # its guidance cleanly instead of a traceback.
        print(f"\nerror: {e}")
        raise SystemExit(1) from e
    finally:
        logger.close()
    print(f"\nTraining finished. Best eval return: "
          f"{result['best_eval_return']:.1f}")
    print(f"Artifacts in: {logdir}")


if __name__ == "__main__":
    main()
