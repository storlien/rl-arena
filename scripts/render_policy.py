"""
Render a trained CartPole policy as an animated GIF + trace plots.

Usage:
    uv run python scripts/render_policy.py runs/cartpole_xxx/best_policy.pt
    uv run python scripts/render_policy.py runs/cartpole_xxx/best_policy.pt \
            --gif myrollout.gif --stochastic

Outputs (next to the checkpoint by default):
    <stem>_rollout.gif        animation of one full 240-step episode
    <stem>_traces.png         pole angle & per-step reward over the episode

Headless-safe (matplotlib Agg backend) -- works over SSH on a cluster;
copy the GIF out or open it in a file browser afterwards.
"""

import argparse
import math
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

import torch

from shac.config import EnvConfig
from shac.envs.cartpole_swingup import CartPoleSwingUpEnv
from shac.models.networks import GaussianActor
from shac.utils.running_stats import RunningMeanStd


def load_policy(path, device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    env_cfg = EnvConfig(**ckpt["env_cfg"])
    # CartPole observation is always 5-dim (Table 6).
    actor = GaussianActor(5, 1,
                          tuple(ckpt["algo_cfg"]["actor_units"]),
                          ckpt["algo_cfg"]["actor_logstd_init"]).to(device)
    actor.load_state_dict(ckpt["actor"])
    actor.eval()
    rms = None
    if ckpt["obs_rms"] is not None:
        rms = RunningMeanStd((5,), device=device)
        rms.load_state_dict(ckpt["obs_rms"])
    return env_cfg, actor, rms


@torch.no_grad()
def rollout(env, actor, rms, deterministic=True):
    """Run ONE full task episode on env (num_envs=1), recording frames."""
    obs = env.reset()
    frames = []          # (x, theta, xdot, thetadot)
    rewards = []
    total = 0.0
    for t in range(env.episode_length):
        o = rms.normalize(obs) if rms is not None else obs
        action = torch.tanh(actor(o, deterministic=deterministic))
        obs, rew, done, _ = env.step(action)
        total += float(rew.item())
        frames.append((float(env.q[0, 0]), float(env.q[0, 1]),
                       float(env.qd[0, 0]), float(env.qd[0, 1])))
        rewards.append(float(rew.item()))
        if done.any():
            break
    return frames, rewards, total


def draw_frame(ax, x, theta, t, total_rew, env):
    ax.clear()
    # rail
    ax.plot([-1.2, 1.2], [0, 0], color="gray", lw=1)
    # cart (0.5 x 0.2 box, like the official URDF visuals)
    ax.add_patch(plt.Rectangle((x - 0.25, -0.1), 0.5, 0.2,
                               fc="#4a7ebb", ec="k"))
    # pole: 1 m long, theta measured from upright
    px, py = x + math.sin(theta), math.cos(theta)
    ax.plot([x, px], [0, py], color="k", lw=4, solid_capstyle="round")
    ax.add_patch(plt.Circle((px, py), 0.045, fc="k"))
    # goal direction marker
    ax.plot(0, 1.15, marker="^", color="green", ms=8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.25, 1.3)
    ax.set_aspect("equal")
    th_n = math.atan2(math.sin(theta), math.cos(theta))
    ax.set_title(f"step {t}/{env.episode_length}   "
                 f"theta = {math.degrees(th_n):6.1f} deg   "
                 f"return so far = {total_rew:8.1f}")
    ax.set_xlabel("x [m]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--gif", default=None, help="output gif path")
    p.add_argument("--traces", default=None, help="output traces png path")
    p.add_argument("--stochastic", action="store_true",
                   help="sample actions instead of using the mean")
    p.add_argument("--device", default="cpu")
    p.add_argument("--fps", type=int, default=30)
    args = p.parse_args()

    device = torch.device(args.device)
    env_cfg, actor, rms = load_policy(args.checkpoint, device)
    env_cfg = EnvConfig(**{**env_cfg.__dict__, "num_envs": 1,
                           "stochastic_init": True})
    env = CartPoleSwingUpEnv(env_cfg, device)

    frames, rewards, total = rollout(env, actor, rms,
                                     deterministic=not args.stochastic)
    print(f"episode return: {total:.1f}  (hanging-still baseline = -2369, "
          f"perfect ~ 0; closer to 0 = better)")

    stem = os.path.splitext(args.checkpoint)[0]
    gif_path = args.gif or stem + "_rollout.gif"
    traces_path = args.traces or stem + "_traces.png"

    # ---------------- GIF ----------------
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    running = [0.0]

    def update(i):
        running[0] += rewards[i]
        draw_frame(ax, *frames[i][0:2], i, running[0], env)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // args.fps)
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved: {gif_path}")

    # ---------------- traces ----------------
    thetas = [math.atan2(math.sin(f[1]), math.cos(f[1])) for f in frames]
    xs = [f[0] for f in frames]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot([math.degrees(t) for t in thetas], lw=1)
    axes[0].axhline(0, color="green", ls=":", label="upright (goal)")
    axes[0].axhline(180, color="gray", ls=":", label="hanging")
    axes[0].set_ylabel("pole angle [deg]")
    axes[0].legend(fontsize=8)
    axes[1].plot(xs, lw=1)
    axes[1].set_ylabel("cart x [m]")
    axes[2].plot(rewards, lw=1, color="C3")
    axes[2].set_ylabel("per-step reward")
    axes[2].set_xlabel("policy step")
    axes[0].set_title(f"Episode traces (return {total:.1f})")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(traces_path, dpi=130)
    print(f"saved: {traces_path}")


if __name__ == "__main__":
    main()
