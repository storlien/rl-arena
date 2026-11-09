"""
Render a trained policy (CartPole or Pendulum) as an animated GIF + traces.

Usage:
    uv run python scripts/render_policy.py runs/cartpole_xxx/best_policy.pt
    uv run python scripts/render_policy.py runs/cartpole_xxx/best_policy.pt \
            --gif myrollout.gif --stochastic
    uv run python scripts/render_policy.py runs/pendulum_xxx/best_policy.pt

The environment is read from the checkpoint ("env_name"; older cartpole-only
checkpoints without the field are treated as cartpole).

Outputs (next to the checkpoint by default):
    <stem>_rollout.gif        animation of one full 240-step episode
    <stem>_traces.png         angle & per-step reward over the episode

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

from shac.envs import make_env, make_env_config
from shac.utils.checkpoints import build_actor_and_rms, read_checkpoint


def load_policy(path, device):
    """Load a checkpoint -> (env_name, env_cfg, ckpt).

    The actor (whose input size is an env property) and the obs normalizer
    are rebuilt in main() once the env exists (shac.utils.checkpoints)."""
    return read_checkpoint(path, device)


@torch.no_grad()
def rollout(env, actor, rms, deterministic=True):
    """Run ONE full task episode on env (num_envs=1), recording frames."""
    obs = env.reset()
    frames = []          # ([q...], [qd...]) per step, env-agnostic
    rewards = []
    total = 0.0
    for t in range(env.episode_length):
        o = rms.normalize(obs) if rms is not None else obs
        action = torch.tanh(actor(o, deterministic=deterministic))
        obs, rew, done, _ = env.step(action)
        total += float(rew.item())
        frames.append(([float(v) for v in env.q[0]],
                       [float(v) for v in env.qd[0]]))
        rewards.append(float(rew.item()))
        if done.any():
            break
    return frames, rewards, total


# ---------------------------------------------------------------------------
# Per-env rendering: drawing and trace labels
# ---------------------------------------------------------------------------
def draw_frame_cartpole(ax, q, t, total_rew, env):
    x, theta = q[0], q[1]
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


def draw_frame_pendulum(ax, q, t, total_rew, env):
    theta = q[0]
    ax.clear()
    # pivot
    ax.add_patch(plt.Circle((0, 0), 0.05, fc="gray", ec="k"))
    # arm: 1 m link, theta measured from upright
    px, py = math.sin(theta), math.cos(theta)
    ax.plot([0, px], [0, py], color="k", lw=4, solid_capstyle="round")
    ax.add_patch(plt.Circle((px, py), 0.09, fc="#4a7ebb", ec="k"))
    # goal direction marker
    ax.plot(0, 1.15, marker="^", color="green", ms=8)
    ax.set_xlim(-1.3, 1.3)
    ax.set_ylim(-1.25, 1.3)
    ax.set_aspect("equal")
    th_n = math.atan2(math.sin(theta), math.cos(theta))
    ax.set_title(f"step {t}/{env.episode_length}   "
                 f"theta = {math.degrees(th_n):6.1f} deg   "
                 f"return so far = {total_rew:8.1f}")
    ax.set_xlabel("y [m]")


DRAWERS = {
    "cartpole": draw_frame_cartpole,
    "pendulum": draw_frame_pendulum,
}

# second trace row: what best shows "what else the policy is doing"
TRACE_LABELS = {
    "cartpole": ("cart x [m]", lambda q, qd: q[0]),
    "pendulum": ("theta_dot [rad/s]", lambda q, qd: qd[0]),
}


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
    env_name, env_cfg, ckpt = load_policy(args.checkpoint, device)
    env_cfg = make_env_config(env_name, **{**env_cfg.__dict__, "num_envs": 1,
                                           "stochastic_init": True})
    env = make_env(env_name, env_cfg, device)

    # Actor shape follows the env (cartpole obs = 5, pendulum obs = 3).
    actor, rms = build_actor_and_rms(ckpt, env, device)

    frames, rewards, total = rollout(env, actor, rms,
                                     deterministic=not args.stochastic)
    # The do-nothing baseline: hanging still scores -pi^2 per step for BOTH
    # envs' wrapped-angle rewards (derived, not hardcoded, so it follows
    # each run's task horizon).
    baseline = -math.pi ** 2 * env.episode_length
    print(f"episode return: {total:.1f}  (hanging-still baseline = "
          f"{baseline:.0f}, perfect ~ 0; closer to 0 = better)")

    stem = os.path.splitext(args.checkpoint)[0]
    gif_path = args.gif or stem + "_rollout.gif"
    traces_path = args.traces or stem + "_traces.png"

    # ---------------- GIF ----------------
    draw = DRAWERS[env_name]
    fig, ax = plt.subplots(figsize=(4.2, 4.6))
    running = [0.0]

    def update(i):
        running[0] += rewards[i]
        draw(ax, frames[i][0], i, running[0], env)

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=1000 // args.fps)
    anim.save(gif_path, writer=PillowWriter(fps=args.fps))
    plt.close(fig)
    print(f"saved: {gif_path}")

    # ---------------- traces ----------------
    def wrapped(q):
        return math.atan2(math.sin(q[1 if env_name == "cartpole" else 0]),
                          math.cos(q[1 if env_name == "cartpole" else 0]))

    thetas = [wrapped(q) for q, _ in frames]
    label, second = TRACE_LABELS[env_name]
    seconds = [second(q, qd) for q, qd in frames]
    fig, axes = plt.subplots(3, 1, figsize=(9, 7), sharex=True)
    axes[0].plot([math.degrees(t) for t in thetas], lw=1)
    axes[0].axhline(0, color="green", ls=":", label="upright (goal)")
    axes[0].axhline(180, color="gray", ls=":", label="hanging")
    axes[0].set_ylabel("angle [deg]")
    axes[0].legend(fontsize=8)
    axes[1].plot(seconds, lw=1)
    axes[1].set_ylabel(label)
    axes[2].plot(rewards, lw=1, color="C3")
    axes[2].set_ylabel("per-step reward")
    axes[2].set_xlabel("policy step")
    axes[0].set_title(f"Episode traces ({env_name}, return {total:.1f})")
    for a in axes:
        a.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(traces_path, dpi=130)
    print(f"saved: {traces_path}")


if __name__ == "__main__":
    main()
