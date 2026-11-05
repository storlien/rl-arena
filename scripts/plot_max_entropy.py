"""
The max-entropy A/B figure (see MAX_ENTROPY.md).

Usage:
    uv run python scripts/plot_max_entropy.py            # -> figures/max_entropy_ab.png
    uv run python scripts/plot_max_entropy.py -o other.png
    uv run python scripts/plot_max_entropy.py --plain-global runs/... ...

Reads the FOUR runs' log.csv (defaults: the MAX_ENTROPY.md arms -- the
two entropy arms of this experiment plus the two plain arms reused from
the state-std A/B) and produces a 2x2 figure:

  (a) deterministic-eval return curves, best-value markers
  (b) policy_std trajectories, log scale -- the entropy FLOOR vs the
      plain arms' decay/collapse
  (c) the dual at work: entropy_alpha and policy_entropy trajectories
      against the target H-bar = -0.5
  (d) actor gradient-estimator variance (the --grad-var-interval probes)

Solid lines are the plain arms (STATE_STD.md), dashed lines the
max-entropy arms; blue = global sigma, red = state sigma(s).
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_training import load_log, col, masked

C_G, C_S = "C0", "C3"          # global = blue, state = red


def smooth(ys, k=5):
    return [sum(ys[max(0, i - k + 1): i + 1]) /
            len(ys[max(0, i - k + 1): i + 1]) for i in range(len(ys))]


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--plain-global", default="runs/std_global_500ep")
    p.add_argument("--plain-state", default="runs/std_state_500ep")
    p.add_argument("--ent-global", default="runs/std_ent_global_500ep")
    p.add_argument("--ent-state", default="runs/std_ent_state_500ep")
    p.add_argument("-o", "--out", default="figures/max_entropy_ab.png")
    args = p.parse_args()

    pg = load_log(args.plain_global)
    ps = load_log(args.plain_state)
    eg = load_log(args.ent_global)
    es = load_log(args.ent_state)
    arms = [
        (pg, "global $\\sigma$ (official SHAC)", C_G, "-"),
        (eg, "global $\\sigma$ + max-entropy", C_G, "--"),
        (ps, "state $\\sigma(s)$", C_S, "-"),
        (es, "state $\\sigma(s)$ + max-entropy", C_S, "--"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # ---------- (a) eval-return learning curves ----------------------------
    ax = axes[0, 0]
    bests = []
    for rows, name, c, ls in arms:
        eps, rets = masked(col(rows, "episode"), col(rows, "eval_return_mean"))
        ax.plot(eps, rets, color=c, lw=1.0, alpha=0.3, ls=ls)
        ax.plot(eps, smooth(rets), color=c, lw=2.0, ls=ls, label=name)
        bests.append((eps[rets.index(max(rets))], max(rets), c, ls))
    # best points as markers (the four bests land in one tight cluster at
    # the right edge -- text there is illegible, so the exact numbers go
    # into the empty region below the curves instead: no eval dips
    # below -1069 after episode 170)
    for ep_b, v, c, ls in bests:
        ax.plot([ep_b], [v], marker="o", ms=5, color=c, ls=ls)
    ax.text(180, -4500, "best eval:", fontsize=8.5, va="top")
    for y, (ep_b, v, c, ls) in zip((-5100, -5700, -6300, -6900),
                                  sorted(bests, key=lambda t: -t[1])):
        ax.text(180, y, f"  {v:.1f}  (ep {ep_b})", fontsize=8.5,
                va="top", color=c)
    ax.set_xlabel("episode")
    ax.set_ylabel("eval return (240 steps, $\\to$ 0 is better)")
    ax.set_title("(a) The entropy objective does NOT beat plain SHAC "
                 "at this budget")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # ---------- (b) policy_std: the entropy floor ---------------------------
    ax = axes[0, 1]
    for rows, name, c, ls in arms:
        ax.semilogy(col(rows, "episode"), col(rows, "policy_std"),
                    color=c, lw=1.7, ls=ls, label=name)
    ax.axhline(math.exp(-1), color="gray", ls=":", lw=1)
    # The band just below the init line is NEVER empty (the state+ent arm
    # stays noisy in 0.18-0.36 through ep 494), but the strip above 0.65
    # is empty past episode ~55 -- label the line from there with an arrow.
    ax.annotate("init $e^{-1}\\approx 0.368$", xy=(250, math.exp(-1)),
                xytext=(250, 1.02), fontsize=8, color="gray",
                ha="center", va="bottom",
                arrowprops=dict(arrowstyle="-", color="gray", lw=0.8))
    ax.set_xlabel("episode")
    ax.set_ylabel("policy std (log scale)")
    ax.set_title("(b) The dual holds $\\sigma$ at the $H=-0.5$ floor "
                 "($\\approx$ 0.15):\n no decay, no collapse")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    # ---------- (c) the dual: alpha and entropy vs the target --------------
    ax = axes[1, 0]
    ax2 = ax.twinx()
    for rows, name, c in ((eg, "global", C_G), (es, "state", C_S)):
        eps, a = masked(col(rows, "episode"), col(rows, "entropy_alpha"))
        ax.plot(eps, a, color=c, lw=2.0, label=f"$\\alpha$ ({name})")
        eps, h = masked(col(rows, "episode"), col(rows, "policy_entropy"))
        ax2.plot(eps, h, color=c, lw=1.2, alpha=0.55)
    ax2.axhline(-0.5, color="k", ls=":", lw=1.2)
    ax2.annotate("target $\\bar{H} = -\\dim(A)/2 = -0.5$",
                 xy=(150, -0.5), xytext=(150, -0.72), fontsize=8,
                 color="k")
    ax.set_xlabel("episode")
    ax.set_ylabel("temperature $\\alpha$ (solid)")
    ax2.set_ylabel("window entropy $h$ (faint)")
    ax2.set_ylim(-1.0, 1.0)
    ax.set_title("(c) SAC's dual works: $\\alpha$ decays 1.0 $\\to$ 0.23 "
                 "and parks $h$ at $\\bar{H}$")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    # ---------- (d) gradient-estimator variance ----------------------------
    ax = axes[1, 1]
    for rows, name, c, ls in arms:
        eps, var = masked(col(rows, "episode"), col(rows, "actor_grad_var"))
        ax.semilogy(eps, var, color=c, lw=1.5, ls=ls, label=name)
    ax.set_xlabel("episode")
    ax.set_ylabel("Var[$\\hat{\\nabla}$] (log scale)")
    ax.set_title("(d) Actor gradient-estimator variance (k=8 probes)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("Max-entropy (SAC/SAPO-style) objective vs plain SHAC -- "
                 "both sigma modes -- CartPole, 500 episodes, seed 0",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved: {args.out}")
    for rows, name, _, _ in arms:
        eps, rets = masked(col(rows, "episode"), col(rows, "eval_return_mean"))
        a = rows[-1].get("entropy_alpha", "")
        h = rows[-1].get("policy_entropy", "")
        print(f"  {name}: best {max(rets):.1f}, final std "
              f"{float(rows[-1]['policy_std']):.4f}"
              + (f", alpha {float(a):.3f}, H {float(h):+.2f}" if a else ""))


if __name__ == "__main__":
    main()
