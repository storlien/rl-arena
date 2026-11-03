"""
Plot the training progression from a run's log.csv.

Usage:
    uv run python scripts/plot_training.py runs/cartpole_20260103_120000
    uv run python scripts/plot_training.py runs/cartpole_20260103_120000 -o my_plot.png
    uv run python scripts/plot_training.py runs/cartpole_20260103_120000 --smooth 0   # raw

Produces a 2x2 figure, with the paper's Figure 4 axes:
  (1) episode return vs simulation steps (x10^5)  (train + eval curves)
  (2) episode return vs wall-clock time            (the paper's other axis)
  (3) actor / critic losses        (also vs simulation steps)
  (4) diagnostics: policy std, gradient norms

ABOUT SMOOTHING: every series is one point per learning episode from a single
seed, which is much noisier than the paper's Figure 4, where each curve is a
mean over 5 seeds (which is what makes the paper's curves smooth). All curves
are therefore drawn with a centered rolling window around a base window of
~4% of the run (20 episodes for a 500-episode run): eval curves use 2x the
base window (their rows are 10x sparser), losses and diagnostics use 4x.
The loss and gradient-norm series are smoothed with a rolling median, which
ignores the periodic task-reset spikes instead of smearing them into the
trend. --smooth N rescales the base window; --smooth 0/1 plots raw
per-episode values.

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
import statistics
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


def smooth(ys, window, stat="mean"):
    """Centered rolling mean (default) or median over a series with None gaps.

    The paper's Figure 4 smooths by averaging 5 independent seeds. This run
    is a single seed recorded once per learning episode, so we use a rolling
    window over the episodes instead -- the standard way to present
    single-seed learning curves. None entries (e.g. eval columns between
    evals) are skipped, so an eval series is averaged over its ~window/10
    neighbours.

    The diagnostic series (actor/critic loss, gradient norm) are spike-
    dominated: every 240/32 = 7.5 episodes all 64 envs reset in lockstep,
    and those windows carry losses orders of magnitude above the typical
    episode. For those we use a rolling *median* (stat="median"): it ignores
    the reset spikes entirely, whereas a mean leaves residual ripple unless
    the window spans ~10 reset periods -- by which point real trends are
    over-smoothed.
    """
    n = len(ys)
    if window is None or window <= 1 or n == 0:
        return ys
    half = max(1, int(window) // 2)
    out = []
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        vals = [y for y in ys[lo:hi] if y is not None]
        if not vals:
            out.append(None)
        elif stat == "median":
            out.append(statistics.median(vals))
        else:
            out.append(sum(vals) / len(vals))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("logdir", help="run directory containing log.csv")
    p.add_argument("-o", "--output", default=None,
                   help="output png (default: <logdir>/training.png)")
    p.add_argument("--smooth", type=int, default=-1,
                   help="centered moving-average window in episodes applied "
                        "to every curve (default: ~4%% of the run; 0 or 1 "
                        "= raw per-episode values)")
    args = p.parse_args()

    rows = load_log(args.logdir)
    window = args.smooth
    if window < 0:                          # auto: ~4% of the run
        window = max(5, len(rows) // 25)
    # Eval rows are 10x sparser than train rows, so eval curves get a 2x
    # wider window. The loss / diagnostic series are dominated by the
    # synchronized task resets (all 64 envs reset in lockstep every
    # 240/32 = 7.5 episodes), so they get a 4x window combined with a
    # rolling median that ignores the reset spikes (see smooth()).
    w_eval = 2 * window
    w_loss = 4 * window
    steps = col(rows, "env_steps")
    wall = col(rows, "wall_time_s")
    # The paper's Figure 4 plots learning curves against "simulation steps
    # (x10^5)" -- total environment steps summed over all parallel envs.
    # Use the same unit here; panel 2 additionally mirrors the paper's
    # bottom-row axis (wall-clock time in minutes).
    steps_1e5 = [s / 1e5 for s in steps]
    wall_min = [w / 60.0 for w in wall]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))
    fig.suptitle("SHAC on CartPole Swing-Up", fontsize=14)

    # ---------------- (1) return vs simulation steps ----------------
    ax = axes[0, 0]
    tr_m, tr_s = col(rows, "train_return_mean"), col(rows, "train_return_std")
    ev_m, ev_s = col(rows, "eval_return_mean"), col(rows, "eval_return_std")
    tr_ms, tr_ss = smooth(tr_m, window), smooth(tr_s, window)
    ev_ms, ev_ss = smooth(ev_m, w_eval), smooth(ev_s, w_eval)
    ex, em = masked(steps_1e5, ev_ms)
    _, es = masked(steps_1e5, ev_ss)
    tx, tm = masked(steps_1e5, tr_ms)
    _, ts = masked(steps_1e5, tr_ss)
    if tm:
        ax.plot(tx, tm, lw=1.8, color="C0", label="train (smoothed)")
        ax.fill_between(tx, [m - s for m, s in zip(tm, ts)],
                        [m + s for m, s in zip(tm, ts)],
                        alpha=0.15, color="C0", lw=0)
    if em:
        ax.plot(ex, em, color="C1", lw=2, marker="o", ms=3,
                label="eval (smoothed)")
        ax.fill_between(ex, [m - s for m, s in zip(em, es)],
                        [m + s for m, s in zip(em, es)],
                        alpha=0.2, color="C1", lw=0)
    ax.axhline(-2369, color="gray", ls=":",
               label="hanging still (-pi^2 * 240 = -2369)")
    ax.axhline(0.0, color="green", ls=":", alpha=0.7, label="perfect (0)")
    ax.set_xlabel("simulation steps ($\\times 10^5$)")
    ax.set_ylabel("episode return  (higher = better)")
    ax.set_title("Return vs simulation steps (cf. paper Fig. 4 top)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # ---------------- (2) return vs wall-clock time ----------------
    # The paper's Figure 4 bottom row: the same learning curve against
    # wall-clock training time instead of samples.
    ax = axes[0, 1]
    wx, wm = masked(wall_min, ev_ms)
    if wm:
        ax.plot(wx, wm, color="C1", lw=2)
    ax.axhline(-2369, color="gray", ls=":")
    ax.set_xlabel("wall-clock time (min)")
    ax.set_ylabel("eval episode return")
    ax.set_title("Return vs wall-clock time (cf. paper Fig. 4 bottom)")
    ax.grid(alpha=0.3)

    # ---------------- (3) losses ----------------
    ax = axes[1, 0]
    al = col(rows, "actor_loss")
    cl = col(rows, "critic_loss")
    ax.plot(steps_1e5, smooth(al, w_loss, stat="median"), color="C2", lw=1.5,
            label="actor loss (Eq. 5, median-smoothed)")
    ax.set_xlabel("simulation steps ($\\times 10^5$)")
    ax.set_ylabel("actor loss", color="C2")
    ax2 = ax.twinx()
    ax2.plot(steps_1e5, smooth(cl, w_loss, stat="median"), color="C3", lw=1.5,
             label="critic MSE loss (Eq. 6, median-smoothed)")
    # critic loss spans ~1e-3 (converged) to ~1e5 (first episodes' fitting
    # spike); log scale keeps both regimes visible.
    ax2.set_yscale("log")
    ax2.set_ylabel("critic loss (log)", color="C3")
    ax.set_title("Losses")
    ax.grid(alpha=0.3)

    # ---------------- (4) diagnostics ----------------
    ax = axes[1, 1]
    sd = col(rows, "policy_std")
    gn = col(rows, "actor_grad_norm")
    ax.plot(steps_1e5, smooth(sd, w_loss), color="C4", lw=1.5,
            label="policy std")
    ax.set_xlabel("simulation steps ($\\times 10^5$)")
    ax.set_ylabel("policy std", color="C4")
    ax2 = ax.twinx()
    ax2.plot(steps_1e5, smooth(gn, w_loss, stat="median"), color="C5", lw=1.5,
             alpha=0.8, label="actor grad norm")
    ax2.set_yscale("log")
    ax2.set_ylabel("actor grad norm (log)", color="C5")
    ax.set_title("Diagnostics")
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = args.output or os.path.join(args.logdir, "training.png")
    fig.savefig(out, dpi=130)
    if window > 1:
        print(f"saved: {out}  (smoothing windows: train {window} eps, "
              f"eval {w_eval} eps, losses/diagnostics {w_loss} eps)")
    else:
        print(f"saved: {out}  (raw per-episode values, no smoothing)")


if __name__ == "__main__":
    main()
