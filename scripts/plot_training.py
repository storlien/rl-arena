"""
Plot the training progression from a run's log.csv.

Usage:
    uv run python scripts/plot_training.py runs/cartpole_20260103_120000
    uv run python scripts/plot_training.py runs/cartpole_20260103_120000 -o my_plot.png

Produces a 2x2 figure:
  (1) episode return vs learning episode   (train + eval curves)
  (2) episode return vs total env steps    (the paper's sample-efficiency axis)
  (3) actor / critic losses
  (4) diagnostics: policy std, gradient norms

ABOUT THE RETURN SIGN (important when comparing with the paper):
The CartPole reward R = -theta^2 - ... is always <= 0, so the episode return
(sum of R over the 240-step task episode) is a NEGATIVE number that
approaches 0 as the policy learns to swing up and balance:
    ~ -2370  pole hanging down the whole episode (do-nothing baseline)
    ->   ~0  pole balanced upright (perfect policy)
So HIGHER (closer to 0) = BETTER here. The paper's Figure 4 CartPole panel
plots the negated quantity (a positive "loss" that descends as training
progresses); both views carry the same information.
"""

import argparse
import csv
import os
import sys

import matplotlib
matplotlib.use("Agg")           # headless-safe (works over SSH)
import matplotlib.pyplot as plt


def load_log(logdir: str):
    path = os.path.join(logdir, "log.csv")
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"empty log: {path}")
    return rows


def col(rows, name, cast=float):
    out = []
    for r in rows:
        v = r.get(name, "")
        if v not in ("", None):
            out.append(cast(v))
        else:
            out.append(None)      # missing (e.g. eval columns between evals)
    return out


def masked(xs, ys):
    """Keep only points where y is not None (for sparse eval columns)."""
    return ([x for x, y in zip(xs, ys) if y is not None],
            [y for y in ys if y is not None])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logdir", help="run directory containing log.csv")
    p.add_argument("-o", "--output", default=None,
                   help="output png (default: <logdir>/training.png)")
    args = p.parse_args()

    rows = load_log(args.logdir)
    ep = col(rows, "episode")
    steps = col(rows, "env_steps")
    wall = col(rows, "wall_time_s")

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("SHAC on CartPole Swing-Up", fontsize=14)

    # ---------------- (1) return vs episode ----------------
    ax = axes[0, 0]
    tr_m, tr_s = col(rows, "train_return_mean"), col(rows, "train_return_std")
    ev_m, ev_s = col(rows, "eval_return_mean"), col(rows, "eval_return_std")
    ex, em = masked(ep, ev_m)
    _, es = masked(ep, ev_s)
    tx, tm = masked(ep, tr_m)
    _, ts = masked(ep, tr_s)
    if tm:
        ax.plot(tx, tm, alpha=0.4, lw=0.8, color="C0",
                label="train (per-episode mean)")
        ax.fill_between(tx, [m - s for m, s in zip(tm, ts)],
                        [m + s for m, s in zip(tm, ts)],
                        alpha=0.15, color="C0", lw=0)
    if em:
        ax.plot(ex, em, color="C1", lw=2, marker="o", ms=3,
                label="eval (deterministic)")
        ax.fill_between(ex, [m - s for m, s in zip(em, es)],
                        [m + s for m, s in zip(em, es)],
                        alpha=0.2, color="C1", lw=0)
    ax.axhline(-2369, color="gray", ls=":",
               label="hanging still (-pi^2 * 240 = -2369)")
    ax.axhline(0.0, color="green", ls=":", alpha=0.7, label="perfect (0)")
    ax.set_xlabel("learning episode")
    ax.set_ylabel("episode return  (higher = better)")
    ax.set_title("Return vs episodes")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---------------- (2) return vs env steps ----------------
    ax = axes[0, 1]
    sx, sm = masked(steps, ev_m)
    if sm:
        ax.plot(sx, sm, color="C1", lw=2)
    ax.axhline(-2369, color="gray", ls=":")
    ax.set_xlabel("total environment steps (sample-efficiency axis)")
    ax.set_ylabel("eval episode return")
    ax.set_title("Return vs simulation steps (cf. paper Fig. 4 top)")
    ax.grid(alpha=0.3)

    # ---------------- (3) losses ----------------
    ax = axes[1, 0]
    al = col(rows, "actor_loss")
    cl = col(rows, "critic_loss")
    ax.plot(ep, al, color="C2", lw=1, label="actor loss (Eq. 5)")
    ax.set_xlabel("learning episode")
    ax.set_ylabel("actor loss", color="C2")
    ax2 = ax.twinx()
    ax2.plot(ep, cl, color="C3", lw=1, label="critic MSE loss (Eq. 6)")
    ax2.set_ylabel("critic loss", color="C3")
    ax.set_title("Losses")
    ax.grid(alpha=0.3)

    # ---------------- (4) diagnostics ----------------
    ax = axes[1, 1]
    sd = col(rows, "policy_std")
    gn = col(rows, "actor_grad_norm")
    ax.plot(ep, sd, color="C4", lw=1.5, label="policy std")
    ax.set_xlabel("learning episode")
    ax.set_ylabel("policy std", color="C4")
    ax2 = ax.twinx()
    ax2.plot(ep, gn, color="C5", lw=1, alpha=0.8, label="actor grad norm")
    ax2.set_yscale("log")
    ax2.set_ylabel("actor grad norm (log)", color="C5")
    ax.set_title("Diagnostics")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.output or os.path.join(args.logdir, "training.png")
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
