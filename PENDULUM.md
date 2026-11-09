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
* Evaluation forks the global torch RNG (SHACTrainer.evaluate uses
  `torch.random.fork_rng`), so eval episodes never perturb the training
  rollout stream — changing `--eval-interval`/`--eval-episodes` does not
  change a training trajectory. Runs trained before this change consumed
  eval draws from the shared RNG: statistically equivalent, but not
  bit-reproducible against the current code.
* The env is unbounded below only through the wrapped angle
  (`|theta_n| <= pi`), so returns are bounded: `[-pi^2*240*(1+0.1*...), 0]`;
  in practice the worst case (hanging still) is ≈ -2369, same scale as the
  CartPole.

## 5. Results

**Status: the success bar is NOT met.** Training is stable and improves
from hanging (−2369) to a consistent "swing most of the way up and
oscillate through top" policy (best eval −1376, ≈ 18.8% of steps within
14° of upright), but it never locks in the balance, and every attempt to
push past the plateau has come back null so far. Single-seed protocol
(seed 0) throughout; the plateau is horizon-invariant, so we treat it as
a property of the task setup, not of one run.

### 5.1 The baseline run (500 episodes, h = 32)

`runs/pendulum_20261109_155434/` — paper CartPole hyperparameters, only
re-deriving what the task forces (§3):

| metric | value |
|---|---|
| best eval return | **−1376.0** (ep 480) |
| mean of last 5 evals | −1455.2 ± 51.2 |
| policy σ (start → end) | 0.364 → 0.169 |
| swing-up fraction (16 det. eval eps) | 18.8% of steps within 14° of upright |
| time to first upright (when reached) | ≈ 227 of 240 steps |
| balance offset (last 60 steps) | ≈ 56° |

Behaviorally: the deterministic policy pumps energy with alternating
full-ish torques, gets the link swinging through the top region, but
arrives there too fast/misaligned to hold — it keeps orbiting the
upright instead of capturing it. `best_policy_rollout.gif` and
`best_policy_traces.png` in the run dir show the orbit;
`training.png` the curve.

A horizon sweep (`runs/pendulum_h{16,64,128}`, best −1360…−1376 each)
shows the plateau is **h-invariant**, and the σ anneal 0.36 → 0.17 is
too — by the time the link reaches the top region reliably, exploration
has shrunk to ~0.17 and the capture maneuver is never sampled enough to
be reinforced.

### 5.2 The maximum-entropy arms (preregistered 2×2)

The σ collapse suggested the SAC/SAPO-style entropy bonus
([MAX_ENTROPY.md](MAX_ENTROPY.md), already implemented and unit-tested).
Before running, we measured the tanh-Gaussian entropy↔σ mapping at
μ = 0 and found the SAPO-default target Ĥ = −0.5 corresponds to
σ ≈ 0.15 — *below* the plateau's already-collapsed σ ≈ 0.17 — so the
2×2 crosses σ mode (global / state) with target (−0.5 default / 0.0
raised, which floors σ ≈ 0.42, above the init 0.36). Hypothesis: the
dual holds exploration wide enough that the capture maneuver gets
sampled and reinforced, breaking the plateau.

500 episodes each, seed 0, otherwise identical to the baseline:

| arm | best eval (ep) | last 5 evals | σ start → end | Ĥ start → end | α end | swing-up |
|---|---|---|---|---|---|---|
| baseline (no entropy) | **−1376.0** (480) | −1455 ± 51 | 0.36 → 0.17 | — | — | 18.8% |
| A1 global, t = −0.5 | −1427.9 (420) | −1531 ± 69 | 0.37 → **0.64** | +0.28 → −0.46 | 0.34 | 0% |
| A2 global, t = 0.0 | −1515.6 (480) | −1603 ± 60 | 0.37 → **0.72** | +0.28 → +0.02 | 0.57 | 0% |
| B1 state, t = −0.5 | −1428.6 (290) | −1557 ± 66 | 0.37 → **0.64** | +0.28 → −0.55 | 0.33 | 0% |
| B2 state, t = 0.0 | −1538.5 (290) | −1632 ± 53 | 0.37 → **0.70** | +0.28 → −0.03 | 0.58 | 0% |

(`runs/pendulum_ent_{global,state}_{t-05,t00}/`, one `training.png`
each; behavioral metrics from `scripts/policy_metrics.py` on each
`best_policy.pt`.)

**The dual works exactly as designed — and that is the problem.** Every
arm's entropy lands on its target (mechanism H1 confirmed again). But
unlike CartPole, where the floor *pins* σ ≈ 0.15, here σ **grows**
0.37 → 0.64–0.72. The reason is the tanh Jacobian: the entropy of the
squashed policy is suppressed by the mean's distance from 0, and the
learned mean parks at large torques (mean |μ| ≈ 1.2, |tanh μ| ≈ 0.84,
where log(1 − tanh²u) ≈ −1.2). At μ = 0, σ = 0.64 would mean Ĥ ≈ +0.63;
on-policy it is −0.46. To hold Ĥ at target against the reward's pull
toward big torques, the dual lets the base σ inflate — and that noise
destroys the precise alternating-torque timing the swing-up needs (the
direct u² tax of the noise is small — the tanh Jacobian shrinks it where
the mean is large — the real cost is behavioral): all four best policies
score **0% swing-up**, strictly worse than the baseline's 18.8%.

Verdict: the entropy premium fails here for the *opposite* reason it
fails on CartPole. CartPole at 1.02M steps has no exploration problem
left (MAX_ENTROPY.md's interpretation 2); the pendulum genuinely has
one — but its exploration is expensive (torque-taxed noise through a
tanh-squashed 1-D action), and paying that premium everywhere to buy a
few more top-region samples costs more than the samples return at this
budget. Same practical conclusion, sharper mechanism — and the
dose-response runs the *opposite* way on the two tasks: on CartPole,
widening the target to 0.0 *helps* (MAX_ENTROPY.md's raised-target
cross-check), here it strictly hurts. Exploration is cheap on one and
expensive on the other.

### 5.3 The horizon question: is 4 s just too short to swing up?

The GIFs show every policy orbiting the top without capturing, so: is
the torque cap the blocker, or the horizon? Both were tested.

**The torque cap is real but faithful.** Gym's spec: u ∈ [−2, 2] N·m
against a gravity torque of m·g·l = 9.81 N·m — 4.9× stronger. Static
lift is impossible (max actuator work over a half-turn, 2·π ≈ 6.3 J,
is a third of the 2·m·g·l = 19.6 J hanging→upright deficit), so the
link MUST be pumped over multiple swings. The balance basin is razor
thin: holding torque m·g·l·sin(θ) exceeds the cap beyond **θ ≈ 11.5°**,
so capture is only possible if the link arrives near-upright AND slow.

**A hand-designed controller locates the binding constraint.** Classic
energy-shaping swing-up (u = k·(E* − E)·ω, clipped) + PD catch, run in
our exact env: pump-up to the top cone takes **231 steps ≈ 3.9 s** —
essentially the whole H = 240 (4 s) episode, leaving a 9-step (0.15 s)
hold. The SAME controller at H = 480 captures and balances for 4.2 s
straight. So at H = 240 capture is *physically* ~impossible; the
baseline's 18.8%-swing-up "orbit" is the best available behavior.

**But doubling the horizon does not fix learning.** Three seeds at
H = 480 (500 eps, else paper config, `runs/pendulum_H480_s{0,1,2}`):
best −1327/−1380/−1330, and the behavioral metrics are IDENTICAL to the
H = 240 plateau — swing-up fraction 18.8% on all seeds, t_up ≈ 224,
balance ≈ 57–60°. The policies reach the top at the same ~3.7 s and
then orbit through the extra 4 s (see
`runs/pendulum_H480_s0/best_policy_rollout.gif`). The task is solvable
at H = 480 — the hand controller proves it — but the capture maneuver
(arriving near-upright AND slow, inside an ~11.5° basin) is never
sampled and reinforced: orbiting is a locally flat optimum whose reward
gradient does not point at the precise deceleration.

**Conclusion.** The H = 240 horizon made capture physically impossible
and the orbit optimal; H = 480 makes capture possible but learning
still walks past it. The remaining blocker is exploration/credit for a
maneuver whose basin is ~11.5° wide — an algorithm problem (or a reward
that prices energy near E*, e.g. shaping toward E* = m·g·l), not a
physics or horizon problem.

### 5.4 Open levers (superseded — see section 6)

* ~~σ₀ init~~: **run, null** (`runs/pend_highsig`, σ₀ = 1.0): identical
  18.8% orbit. Exploration width was never the blocker.
* ~~Multi-seed confirmation~~: done where it mattered (section 6).
* Longer budget: superseded by the task-spec fix below.

## 6. Resolving the plateau: the task spec was the bug, not the algorithm

The question this section answers: the CartPole winners (V-style,
V+h64, full SAPO) and the §5.3 diagnosis pointed away from the
optimizer — what does it actually take to learn swing-up AND balance?

### 6.1 Phase A — the CartPole winners do not transfer (predicted)

Ported at H = 240, hanging init, seed 0:

| arm | best | behavioral metrics | verdict |
|---|---|---|---|
| baseline | −1376.0 | 18.8% swing-up, t_up 227, 55.9° | plateau |
| SHAC + V-style (`--activation silu --optimizer adamw`) | −1334.8 | 18.8%, t_up 227, 56.4° | same orbit |
| SHAC + V-style + h64 | −1334.6 | 25.0%, t_up 224, 50.1° | same orbit |
| full SAPO (all 5 choices) | −1440.3 | **0%** swing-up | entropy tax kills the pump |

Exactly the CartPole lesson generalized: changes to the *learning
machinery* buy speed on tasks already being solved, nothing on tasks
that stall — and full SAPO's entropy is *actively harmful* on a
timing-critical task (as in the §5.2 2×2).

### 6.2 Phase B — attacking the real blocker: the capture basin

The §5.3 diagnosis: capture requires arriving near-upright AND slow,
inside an ~11.5° basin that hanging-init episodes essentially never
sample. Three preregistered arms:

| arm | change | prediction | result (seed 0) |
|---|---|---|---|
| `pend_upinit` | near-upright init θ~U(−π/4, π/4), ω~U(−0.5, 0.5) (`--init-mode upright`) | balance learned densely; swing-up untouched | **87.5% swing-up, 7.2° balance, best −180** on its init dist — balancing IS learnable |
| `pend_enshape1` | energy-shaping penalty `r −= ((E−E*)/2mgl)²` (`--energy-shaping 1`) | orbit converts to capture attempts | **null** — 25% swing-up orbit, worse returns. The quadratic energy price adds no capture precision |
| `pend_highsig` | σ₀ 0.37 → 1.0 (`--actor-logstd-init 0`) | wider exploration finds the basin | **null** — identical 18.8% orbit |
| `pend_upinit_ft` | warm-start from `pend_upinit` best, fine-tune hanging | balance skill transfers | **fail** — skill decays without gradient support (12.5%, 57° orbit) |

The upright-init probe is the load-bearing result: **the balance
controller is learnable**; the hanging-init plateau simply never
reinforces it. Naive two-stage fails because the fine-tune phase gives
the balance skill zero gradient — all-or-nothing init distributions.

### 6.3 The fix: gym's own init distribution (uniform over the circle)

Our init choice was the repo's CartPole-style convention (§1 justified
it as "every episode a genuine swing-up"). But gym's Pendulum-v1
samples **θ ~ U(−π, π), ω ~ U(−1, 1)** — which is a *built-in
curriculum*: every episode starting in/near the capture basin
(~6.5% strictly inside ±11.5°, more within catchable range) gives the
balance skill dense gradient, while far starts still demand the pump.
`--init-mode uniform` restores exactly gym's distribution:

![The three initial-state distributions](figures/pendulum_init_distributions.png)

| run (500 eps, H=240) | best | uniform-dist behavioral metrics |
|---|---|---|
| `pend_gyminit` seed 0 | −474.7 | 68.8% swing-up, t_up 95, **3.7°** balance |
| seed 1 | −531.9 | 68.8%, t_up 94, 2.9° |
| seed 2 | −511.5 | 68.8%, t_up 88, 18.7° |
| + V-style seed 0 | −477.5 | 68.8% — V-style adds nothing here |

**But from hanging starts these policies do NOT solve swing-up**:
reach-and-hold 16.7–27.1% (baseline: 27.1%), arriving at t ≈ 227 of
240 — the pump still consumes the whole 4 s episode (§5.3). Uniform
init taught capture; the clock still forbids the full maneuver.

### 6.4 Uniform init + the gym-spirit horizon: SOLVED

`--init-mode uniform --ep-length 480` (8 s ≈ gym's 10 s at this repo's
dt; the earlier hanging-init H=480 runs showed the horizon alone does
nothing — both ingredients are needed):

| run | best | uniform-dist metrics | **hanging-start eval** (48 eps) |
|---|---|---|---|
| seed 0 | −596.8 | 100% swing-up, t_up 110 | **100% reach+hold**, t_up 311/480, **0.1°** balance |
| seed 1 | −630.8 | 100%, t_up 109 | **100%**, 299/480, 0.1° |
| seed 2 | −413.1 | 100%, t_up 108 | **100%**, 295/480, 0.1° |

Every seed solves the FULL task from genuine hanging starts: pump
(~5 s), capture, then hold at **0.1°** — tighter than the
hand-designed energy-shaping controller of §5.3 (which held 4.2 s at
H=480). See
`runs/pend_gyminit_H480_s0/hangingstart_rollout.gif`.

### 6.5 What this does and doesn't claim

* **The plateau was a task-spec artifact.** Plain SHAC (paper config,
  no modifications) solves pendulum swing-up once the env spec matches
  gym's — uniform-over-the-circle init + a ~10 s clock. Our two
  "deliberate divergences from gym" (§1) were precisely the two things
  that made the task unlearnable at this budget.
* **The success bar is redefined behaviorally.** The old bar (last-5
  eval ≥ −100, H=240 hanging evals) is unreachable under ANY spec
  change: hanging starts cost ~1700+ return at H=480 before
  accounting for the pump, and uniform evals mix start difficulties.
  The honest criterion — which the winners now meet on every seed — is
  100% reach-and-hold from hanging starts with a tight balance.
* **What did NOT work** is part of the result: energy shaping (null),
  σ₀ widening (null), naive two-stage transfer (skill decay), and the
  entire SAPO-inspired machinery (null-to-harmful here). The lever was
  neither the optimizer, the objective, the exploration width, nor the
  reward — it was the task's init distribution and clock.
* Curriculum insight worth keeping: a mixed init distribution acts as
  a free curriculum for unstable-equilibrium tasks; an all-hanging
  init makes the balancing sub-skill unobservable, and an all-upright
  init makes it untransferable.

### 6.6 Reproducing section 6

```bash
# phase A: the CartPole winners, ported unchanged (H=240, hanging init)
uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed 0 \
    --activation silu --optimizer adamw
uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed 0 \
    --activation silu --optimizer adamw --horizon 64
uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed 0 \
    --max-entropy --state-std --critic-ensemble 2 --no-target-critic \
    --actor-bootstrap-pool avg --activation silu --optimizer adamw --grad-clip 0.5

# phase B: anti-plateau arms
uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed 0 \
    --init-mode upright                        # balance-only curriculum
uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed 0 \
    --energy-shaping 1.0                       # null
uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed 0 \
    --actor-logstd-init 0.0                    # null
uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed 0 \
    --resume runs/pend_upinit/best_policy.pt   # naive two-stage: fails

# the fix (3 seeds): gym init + gym-spirit horizon
for s in 0 1 2; do
  uv run python -m shac.train --env pendulum --device cpu --threads 1 --seed $s \
      --init-mode uniform --ep-length 480
done

# hanging-start honesty check + figures
uv run python scripts/policy_metrics.py runs/pend_gyminit_H480_s0/best_policy.pt
uv run python scripts/plot_training.py runs/pend_gyminit_H480_s0
```
