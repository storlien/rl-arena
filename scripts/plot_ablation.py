"""
Overlay the ablation arms' evaluation curves (mean +/- std across seeds).

Usage:
    uv run python scripts/plot_ablation.py                 # -> figures/ablation_eval.png
    uv run python scripts/plot_ablation.py -o other.png
    uv run python scripts/plot_ablation.py --smooth 5      # opt-in smoothing of
                                                           # the mean curves

Reads the run matrix  runs/ablation/{a_shac_s*, b_nocritic_s*,
c_bptt240_s*, d_h64_s*}/log.csv  (see ABLATION.md for the arm definitions)
and produces a 1x2 figure:

  (left)  deterministic eval return vs simulation steps (x10^5)
  (right) the same vs wall-clock minutes -- the ablation's compute-cost axis

Per arm: thick line = mean across seeds, band = +/-1 std; thin lines are the
individual seeds (drawn to their full extent, so a run that diverged and
died early is still visible where the mean band stops).
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_training import load_log, col, masked, smooth

# arm label -> (run-directory prefix, color)
ARMS = [
    ("A: SHAC (h=32, critic)",            "a_shac_s",     "C0"),
    ("B: no critic (h=32)",               "b_nocritic_s", "C3"),
    ("C: no critic, full BPTT (h=240)",   "c_bptt240_s",  "C2"),
    ("D: no critic (h=64)",               "d_h64_s",      "C6"),
]
NUM_SEEDS = 5


def load_arm(base, prefix):
    """Return list of (steps_1e5, eval_mean, wall_min) per available seed."""
    seeds = []
    for s in range(NUM_SEEDS):
        logdir = os.path.join(base, f"{prefix}{s}")
        if not os.path.exists(os.path.join(logdir, "log.csv")):
            print(f"  warning: missing {logdir} (seed skipped)")
            continue
        rows = load_log(logdir)
        steps = [float(r["env_steps"]) / 1e5 for r in rows]
        wt = [float(r["wall_time_s"]) / 60.0 for r in rows]
        em_full = col(rows, "eval_return_mean")
        ex, em = masked(steps, em_full)
        wt_eval, _ = masked(wt, em_full)    # wall-clock at the eval rows
        seeds.append((np.array(ex), np.array(em), np.array(wt_eval)))
    return seeds


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--base", default="runs/ablation",
                   help="root of the run matrix (default: runs/ablation)")
    p.add_argument("-o", "--output", default="figures/ablation_eval.png")
    p.add_argument("--smooth", type=int, default=0,
                   help="eval-point moving-average window applied to the "
                        "mean curve (default 0 = raw; the 5-seed average "
                        "is already smooth, and smoothing biases the "
                        "endpoint down when the trend improves)")
    args = p.parse_args()

    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12.5, 4.8))
    fig.suptitle("SHAC no-critic ablation -- CartPole Swing-Up "
                 "(5 seeds per arm, ~1.02M env steps per run)", y=0.99)
    footnotes = []

    for label, prefix, color in ARMS:
        seeds = load_arm(args.base, prefix)
        if not seeds:
            print(f"  warning: no runs for {label}")
            continue
        # seeds that diverged before their first evaluation have no eval
        # rows; they cannot join the mean band but are reported below
        eval_seeds = [s for s in seeds if len(s[1]) > 0]
        skipped = len(seeds) - len(eval_seeds)
        if skipped:
            print(f"  note: {label}: {skipped} seed(s) died before the "
                  f"first evaluation -- excluded from the mean band")
            footnotes.append(f"{label[0]}: {skipped} of {len(seeds)} seeds "
                             f"diverged before its first evaluation")
        if not eval_seeds:
            continue
        n = min(len(s[1]) for s in eval_seeds)     # common prefix
        if args.smooth and args.smooth > 1:
            per_seed = [np.array(smooth(s[1].tolist(), args.smooth))[:n]
                       for s in eval_seeds]
        else:
            per_seed = [s[1][:n] for s in eval_seeds]
        ys = np.stack(per_seed)                   # [S, n]
        m, sd = ys.mean(0), ys.std(0, ddof=1)
        xs = eval_seeds[0][0][:n]
        wt = np.stack([s[2][:n] for s in eval_seeds]).mean(0)

        # individual seeds (thin), then the mean band on top
        for sx, sy, _ in eval_seeds:
            ax.plot(sx, sy, color=color, lw=0.7, alpha=0.25)
        ax.plot(xs, m, color=color, lw=2.0, label=label)
        ax.fill_between(xs, m - sd, m + sd, color=color, alpha=0.15, lw=0)
        ax2.plot(wt, m, color=color, lw=2.0, label=label)
        ax2.fill_between(wt, m - sd, m + sd, color=color, alpha=0.15, lw=0)

        best = [float(s[1].max()) for s in eval_seeds]
        print(f"{label:34s} seeds={len(seeds)}  "
              f"best eval {np.mean(best):8.1f} +/- {np.std(best, ddof=1):6.1f}  "
              f"final eval {m[-1]:8.1f} +/- {sd[-1]:6.1f}  "
              f"wall {wt[-1]:5.1f} min")

    for a in (ax, ax2):
        # -2369 = -pi^2 * 240: this script is specific to the CartPole
        # ablation matrix (see ARMS above), whose hanging-still baseline is
        # -pi^2 per step over H = 240 steps. (plot_training.py derives the
        # same quantity from each run's metadata; here the task is fixed.)
        a.axhline(-2369, color="gray", ls=":", lw=1)
        a.axhline(0, color="green", ls=":", lw=1)
        a.set_ylabel("deterministic eval return")
        # returns on this task live in [-2369, 0]; the diverged arm-C seed
        # gives its band a std of thousands, so without a cap the axis
        # would stretch to +5000 and squash the informative region
        a.set_ylim(-2600, 30)
        a.grid(alpha=0.3)
    ax.set_xlabel("simulation steps ($\\times 10^5$)")
    ax2.set_xlabel("wall-clock time (min)")
    ax.set_title("vs simulation steps (cf. paper Fig. 4 top)")
    ax2.set_title("vs wall-clock time (cf. paper Fig. 4 bottom)")
    # label the reference line in the RIGHT panel: its far-right region
    # below the line (t > 1 min) is empty -- only arm A runs that long,
    # and its curve sits far above the line (C/B/D end by ~0.6 min)
    ax2.annotate("hanging still (~-2369)", xy=(0.99, -2378),
                  xycoords=("axes fraction", "data"), ha="right", va="top",
                  fontsize=8, color="gray")
    ax.legend(loc="upper left", fontsize=9)
    if footnotes:
        fig.text(0.5, 0.905, "; ".join(footnotes)
                 + "  (arm C band also clips above 0: one seed diverged "
                   "to ~-13k, inflating its std)",
                 ha="center", fontsize=8, color="dimgray")

    fig.tight_layout(rect=[0, 0, 1, 0.90])
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fig.savefig(args.output, dpi=130)
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()
