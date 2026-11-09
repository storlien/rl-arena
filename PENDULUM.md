# The Pendulum Swing-Up environment (`--env pendulum`)

A second differentiable environment for this repo. **The SHAC paper has no
pendulum task** — its benchmark suite is CartPole, Hopper, Adroit, and
candidates — so this env is *our* extension, added as a second, structurally
different swing-up task (one rotational DoF, torque-limited, no cart) to
check that the implementation is task-agnostic and not accidentally tuned
around the CartPole.

## 1. Task definition and conventions

The TASK spec follows the classic **Gym `Pendulum-v1`** conventions; the
SIMULATION spec follows this repo's CartPole conventions. One from each
column:

| Aspect | Choice | Source |
|---|---|---|
| Angle convention | `theta = 0` **upright**, `theta = +/-pi` hanging | gym (same as the repo's CartPole) |
| Observation (3) | `[cos theta, sin theta, theta_dot]` | gym (cos/sin: continuous at the wrap, same trick as CartPole's sin/cos) |
| Action (1) | torque at the pivot, `u = 2.0 * tanh(a)` N m | gym's +/-2 N m actuator limit |
| Reward | `R = -theta^2 - 0.1*thd^2 - 0.001*u^2` (theta wrapped; u = physical torque) | gym |
| Link | point mass `m = 1` kg, COM `l = 1` m from pivot, `I = m*l^2 = 1` kg m^2, `g = 9.81` | gym |
| Friction / limits | none (continuous hinge, no damping) | gym |
| Integrator | semi-implicit Euler, `dt = 1/60` s per policy step, 4 substeps | **repo** (gym: RK4, dt = 0.05 s) |
| Horizon | `H = 240` steps (4 s), no early termination | **repo** (gym: 200) |
| Initial state | hanging (`theta = -pi`, rest) + uniform noise: `theta += (pi/2)(U-0.5)`, `thd += 0.25(U-0.5)`, gated by `stochastic_init` | **repo** (gym: uniform `theta ~ U[-pi,pi]`, `thd ~ U[-1,1]`) |

Two deliberate divergences from gym deserve justification:

* **Initial state**: gym samples the angle uniformly over the whole circle,
  so a random episode *starts* near the goal roughly 30% of the time, which
  dilutes the "swing-up" in swing-up. The repo's CartPole starts (near)
  hanging with the same noise shape, so we mirror it: every episode is a
  genuine swing-up, and the two envs' initial-state distributions are
  directly comparable.
* **Horizon / dt**: keeping the CartPole's 240 steps @ 60 Hz (with 4 physics
  substeps) makes episodes, buffer shapes, and hyperparameters identical
  across envs; gym's 200 steps @ 25 Hz would change every step count for no
  scientific benefit.

Reward sanity anchors: balanced upright, still, zero torque → `R ≈ 0`
(maximum); hanging still → `R = -pi^2 ≈ -9.87` per step → episode return
`-pi^2 * 240 ≈ -2369` (the same "hanging-still baseline" as the CartPole,
coincidentally). The torque penalty matters during the swing: the classic
failure mode is banging the actuator at ±2 N m, which costs
`0.001 * 4 = 0.004` per step — small, but it breaks symmetric bang-bang ties
in favor of efficient energy pumping.

## 2. Dynamics

One generalized coordinate `theta` (from upright), velocity `thd`. With the
link's COM at height `l*cos(theta)` above the pivot, the Lagrangian is

$$
\mathcal{L} = \tfrac{1}{2} I\,\dot\theta^2 - m g l \cos\theta ,
\qquad I = I_{\text{com}} + m l^2
$$

and the Euler–Lagrange equation with generalized torque `u` gives the whole
dynamics (no mass matrix to invert — the CartPole's 2x2 solve collapses to
one line):

$$
I\,\ddot\theta = u + m g l \sin\theta
\qquad\Rightarrow\qquad
\ddot\theta = \frac{u + m g l \sin\theta}{I}
$$

**Sign check** (this env's tests caught exactly this bug during
development): with `theta` measured from upright, gravity's generalized
torque is `+mgl·sin(theta)` — it pushes *away* from upright (unstable:
`sin(theta) ≈ theta` gives anti-restoring `+omega^2 theta`) and restores
*toward* hanging (`sin(pi + eps) ≈ -eps`). The same sign appears in the
CartPole's hinge equation, and in Gym Pendulum-v1's
`thddot = 3g/(2l) sin(theta) + 3u/(m l^2)` (a *rod*, hence `3/(2l)`:
their `I = ml^2/3`, COM at `l/2`). With the wrong (minus) sign the env
looks superficially plausible but energy is not conserved:
`dE/dt = theta_dot * (I*thddot - mgl sin theta) != 0` for
`E = ½ I thd^2 + mgl cos(theta)`.

A 1-DoF subtlety worth knowing for test-writing: a pendulum released *at
rest* slightly off upright **cannot fall** — its energy caps it at
`|theta| <= theta_0`, since falling over the top would require crossing the
potential maximum at `theta = 0` (barrier height `2mgl` between the two
falling directions). The CartPole's "displace and watch it diverge" test
only works there because the free cart routes energy around the barrier.
The pendulum's instability is therefore tested with a small push, and its
energy conservation unforced (drift measured: ~0.01% over 4 s).

## 3. Hyperparameters (chosen, not from the paper)

The SHAC paper's Tables 4/5 only cover its own tasks. We start from the
**CartPole column** unchanged — same dense-reward swing-up structure — and
only re-derive what the task forces us to:

| Symbol | Meaning | Value | Source |
|---|---|---|---|
| N | parallel envs | 64 | paper Table 4 (CartPole) |
| h | short horizon | 32 | paper Table 4 (CartPole) |
| M | learning episodes | 500 | paper Table 4 (CartPole) |
| H | task horizon | 240 | repo convention (see §1) |
| gamma / lambda | discount / TD(lambda) | 0.99 / 0.95 | paper Table 4 |
| actor / critic lr | 1e-2 / 1e-3 → 1e-5 linear | paper Table 4 (CartPole) | |
| alpha | Polyak | 0.2 | paper Table 4 (CartPole) |
| betas / clip | Adam (0.7, 0.95), grad-norm 1.0 | paper Table 4 | |
| MLPs | [64, 64], ELU + LayerNorm | paper Table 1 | |
| obs dim | 3 (vs CartPole's 5) | task | |
| entropy target | -0.5 (if `--max-entropy`) | `-dim(A)/2` default, dim(A)=1 | |

Success bar agreed before the run: deterministic-policy eval return
averaged over the last 5 evals ≥ **-100** (≈ swings up within ~1–2 s, then
balances with small torque), eval std < 50, plus a visual rollout check.

## 4. Fidelity notes

* Not a SHAC paper task — nothing here can be "faithful to the paper"; the
  claim is only that the *algorithm* runs unmodified and the env satisfies
  the paper's differentiability requirement (smooth `F(s, a)`, smooth
  reward — Eq. 1 and Section 3.1's differentiability assumption).
* Gym `Pendulum-v1` integrates with RK4 at dt = 0.05 and has no substeps;
  we integrate with semi-implicit Euler at dt = 1/60 with 4 substeps to
  match the repo. Episode returns are therefore NOT directly comparable to
  published gym numbers (different horizon length and integration error).
* The torque penalty uses the *physical* torque (`action_strength * a`),
  so the reward is invariant to rescaling the action strength; gym does
  the same.
* The env is unbounded below only through the wrapped angle
  (`|theta_n| <= pi`), so returns are bounded: `[-pi^2*240*(1+0.1*...), 0]`;
  in practice the worst case (hanging still) is ≈ -2369, same scale as the
  CartPole.

## 5. Results

*(to be filled after the 500-episode run — training curve, rollout GIF,
and the behavioral metrics from `scripts/policy_metrics.py`)*
