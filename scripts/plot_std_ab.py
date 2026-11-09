"""
The state-dependent-std A/B overlay figure (see STATE_STD.md).

Usage:
    uv run python scripts/plot_std_ab.py                 # -> figures/state_std_ab.png
    uv run python scripts/plot_std_ab.py -o other.png
    uv run python scripts/plot_std_ab.py --global-run runs/other_global \
                                             --state-run runs/other_state

Reads the two A/B runs' log.csv (defaults: runs/std_global_500ep and
runs/std_state_500ep -- the STATE_STD.md arms) and their best_policy
checkpoints, and produces a 2x2 figure:

  (a) deterministic-eval return curves (raw + smoothed), best-value markers
  (b) policy_std trajectories, log scale -- the exploration-collapse panel
  (c) the learned sigma(s) over a (theta, theta_dot) grid for the state
      arm, with the global arm's constant for reference
  (d) actor gradient-estimator variance (the --grad-var-interval probes)

In state mode policy_std is the window-mean of sigma(s) (see
shac/models/networks.py: GaussianActor.std); in global mode it is the
learnable parameter.
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
import torch
from matplotlib.colors import LogNorm

from plot_training import load_log, col, masked

from shac.envs import make_env
from shac.utils.checkpoints import build_actor_and_rms, read_checkpoint

C_G, C_S = "C0", "C3"          # global = blue, state = red


def load_policy(run, suffix="best_policy"):
    """Rebuild the actor + obs normalizer exactly as the checkpoint was
    trained. The env (and thus actor input size and normalizer shape) comes
    from the checkpoint's env_name via the registry; the state_std flag
    from the algo_cfg metadata."""
    env_name, env_cfg, ckpt = read_checkpoint(os.path.join(run, f"{suffix}.pt"),
                                              "cpu")
    env = make_env(env_name, env_cfg, torch.device("cpu"))
    actor, rms = build_actor_and_rms(ckpt, env, "cpu")
    return actor, rms, env_name


def sigma_grid(env_name):
    """Observation grid over the (theta, theta_dot) plane for panel (c).

    The observation layouts differ per env (cartpole: [x, x_dot, sin, cos,
    theta_dot] with the cart parked at x = x_dot = 0; pendulum:
    [cos, sin, theta_dot]), so the grid builder is dispatched on env_name.
    Returns (obs [n*n, obs_dim], TH, THD) with TH in [0, 2pi] x THD in
    [-6, 6] rad/s.
    """
    th = torch.linspace(0, 2 * math.pi, 121)
    thd = torch.linspace(-6, 6, 61)
    TH, THD = torch.meshgrid(th, thd, indexing="ij")
    if env_name == "cartpole":
        obs = torch.stack([torch.zeros_like(TH).reshape(-1),
                           torch.zeros_like(TH).reshape(-1),
                           TH.sin().reshape(-1), TH.cos().reshape(-1),
                           THD.reshape(-1)], dim=1)
    else:                                   # pendulum
        obs = torch.stack([TH.cos().reshape(-1), TH.sin().reshape(-1),
                           THD.reshape(-1)], dim=1)
    return obs, TH, THD


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--global-run", default="runs/std_global_500ep",
                   help="logdir of the global-sigma arm")
    p.add_argument("--state-run", default="runs/std_state_500ep",
                   help="logdir of the state-sigma arm")
    p.add_argument("-o", "--out", default="figures/state_std_ab.png")
    args = p.parse_args()

    g = load_log(args.global_run)
    s = load_log(args.state_run)
    ge, gx = masked(col(g, "episode"), col(g, "eval_return_mean"))
    se, sx = masked(col(s, "episode"), col(s, "eval_return_mean"))

    fig, axes = plt.subplots(2, 2, figsize=(13, 8.5))

    # ---------- (a) eval-return learning curves ---------------------------
    ax = axes[0, 0]
    for eps, rets, name, c in ((ge, gx, "global $\\sigma$ (official SHAC)", C_G),
                               (se, sx, "state-dependent $\\sigma(s)$ (SAC-style)", C_S)):
        ax.plot(eps, rets, color=c, lw=1.2, alpha=0.35)
        k = 5      # smoothed over 5 evals = 50 episodes
        sm = [sum(rets[max(0, i - k + 1): i + 1]) /
              len(rets[max(0, i - k + 1): i + 1]) for i in range(len(rets))]
        ax.plot(eps, sm, color=c, lw=2.2, label=name)
    best_g, best_s = max(gx), max(sx)
    ax.axhline(best_g, color=C_G, ls=":", lw=1, alpha=0.7)
    ax.axhline(best_s, color=C_S, ls=":", lw=1, alpha=0.7)
    ax.annotate(f"best {best_g:.0f}", xy=(eps[-1], best_g),
                xytext=(380, best_g + 60), fontsize=9, color=C_G)
    ax.annotate(f"best {best_s:.0f}", xy=(eps[-1], best_s),
                xytext=(380, best_s - 200), fontsize=9, color=C_S)
    ax.set_xlabel("episode")
    ax.set_ylabel("eval return (240 steps, $\\to$ 0 is better)")
    ax.set_title("(a) Deterministic-eval return: global $\\sigma$ wins the budget")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    # ---------- (b) policy_std trajectories --------------------------------
    ax = axes[0, 1]
    ax.semilogy(col(g, "episode"), col(g, "policy_std"), color=C_G, lw=1.8,
                label="global $\\sigma$ (one parameter)")
    ax.semilogy(col(s, "episode"), col(s, "policy_std"), color=C_S, lw=1.8,
                label="mean $\\sigma(s)$ (window mean, state mode)")
    ax.axhline(math.exp(-1), color="gray", ls=":", lw=1)
    ax.annotate("init $e^{-1}\\approx 0.368$", xy=(500, math.exp(-1)),
                xytext=(390, 0.43), fontsize=8, color="gray")
    ax.set_xlabel("episode")
    ax.set_ylabel("policy std (log scale)")
    ax.set_title("(b) Exploration width: state mode collapses $\\sigma(s)$ "
                 "$\\sim$ 100$\\times$ harder")
    ax.legend(fontsize=9, loc="lower left")
    ax.grid(alpha=0.3, which="both")

    # ---------- (c) learned sigma(s) over (theta, theta_dot) ---------------
    ax = axes[1, 0]
    actor, rms, env_name = load_policy(args.state_run)
    grid, TH, THD = sigma_grid(env_name)
    # The meshgrid axes for plotting: theta on y, theta_dot on x.
    th = torch.linspace(0, 2 * math.pi, TH.shape[0])
    thd = torch.linspace(-6, 6, THD.shape[1])
    with torch.no_grad():
        sg = actor.mu_sigma(rms.normalize(grid))[1].reshape(TH.shape)
    actor_g, rms_g, _ = load_policy(args.global_run)
    with torch.no_grad():
        sg_g = actor_g.mu_sigma(rms_g.normalize(grid))[1].reshape(TH.shape)
    # vmax includes the global constant so its contour renders INSIDE the
    # colored region (0.0576 > max sigma(s) = 0.036 would draw nothing).
    vmax = max(sg.max().item(), float(sg_g.mean()))
    im = ax.pcolormesh(thd, th, sg, cmap="viridis",
                       norm=LogNorm(vmin=sg.min().item(), vmax=vmax),
                       shading="auto")
    cb = fig.colorbar(im, ax=ax)
    cb.set_label("$\\sigma(s)$ (log color scale)")
    # The global sigma is a CONSTANT 0.0576 at every state (contouring a
    # constant field is degenerate): state it in the annotation instead.
    ax.annotate(f"global $\\sigma$ = {sg_g.mean():.4f} (one constant):\n"
                "above every $\\sigma(s)$ the state mode learned",
                xy=(0.02, 0.03), xycoords="axes fraction", fontsize=9,
                color="white")
    ax.set_xlabel("theta_dot (rad/s)")
    ax.set_ylabel("theta (rad)")
    ax.set_yticks([0, math.pi / 2, math.pi, 3 * math.pi / 2, 2 * math.pi])
    ax.set_yticklabels(["0 (up)", "pi/2", "pi (down)", "3pi/2", "2pi (up)"])
    ax.set_title("(c) Learned $\\sigma(s)$ (state mode): near-deterministic "
                 "upright,\n loudest mid-swing -- vs the global constant")

    # ---------- (d) gradient-estimator variance -----------------------------
    ax = axes[1, 1]
    for rows, name, c in ((g, "global $\\sigma$", C_G),
                          (s, "state $\\sigma(s)$", C_S)):
        eps, var = masked(col(rows, "episode"), col(rows, "actor_grad_var"))
        ax.semilogy(eps, var, color=c, lw=1.6, marker="o", ms=3, label=name)
    ax.set_xlabel("episode")
    ax.set_ylabel("Var[$\\hat{\\nabla}$] (log scale)")
    ax.set_title("(d) Actor gradient-estimator variance (k=8 probes):\n"
                 "the $\\sigma(s)$ head injects pure-noise gradient early")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3, which="both")

    fig.suptitle("A/B: global learnable $\\sigma$ vs SAC-style "
                 "state-dependent $\\sigma(s)$ -- SHAC CartPole, "
                 "500 episodes, seed 0", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=130)
    print(f"saved: {args.out}")
    print(f"sigma(s) grid: min {sg.min():.5f} max {sg.max():.5f} "
          f"(ratio {sg.max()/sg.min():.0f}x); global {sg_g.mean():.5f}")


if __name__ == "__main__":
    main()
