# Experiment: the SAC/SAPO-style maximum-entropy objective

A companion document to the [README](README.md) — the natural third arm
of the [state-dependent σ(s) experiment](STATE_STD.md): SAC couples
σ(s) with an entropy bonus for a reason, so give SHAC the bonus too and
see if it beats the official global σ. Implemented the way SAPO (Xing,
Luk & Oh, ICLR 2025) instantiates SAC's objective inside a short-horizon
window, measured on CartPole Swing-Up at the paper's 500-episode
(~1.02M-step) budget.

**TL;DR.** The objective works mechanically EXACTLY as designed — the
temperature dual decays α 1.0 → 0.23 and parks the policy entropy at the
target Ĥ = −0.5, which floors σ at ≈ 0.15 (the equilibrium
h = −0.5 ⇔ σ ≈ 0.15 predicted in STATE_STD.md) and completely rescues
the state mode's exploration collapse (final window-mean σ(s): 0.0006
→ 0.142). But it does NOT beat the plain global σ at this budget:
best eval **−308.6** (state σ(s) + entropy) and **−314.3** (global σ +
entropy) vs **−274.0** (plain SHAC). At the paper's CartPole budget the
entropy insurance premium costs more than it pays. Verdict: keep the
official objective; the negative result bounds SAPO's scope rather than
contradicting it (their gains are on contact-rich/deformable tasks
where SHAC stalls in local minima — CartPole at 1.02M steps isn't
that).

---

## The question

STATE_STD.md ended with a diagnosis and a promise: σ(s) collapses
~100× harder than the global σ *because SHAC's loss has no entropy
term* — the counter-pressure that keeps SAC's state-dependent σ alive.
SAC's fix is the maximum-entropy objective: augment the return with
`α·H[π]` and auto-tune α to a target entropy. SAPO (ICLR 2025)
transplanted exactly this into SHAC and reports large gains on
differentiable-multiphysics tasks. So two hypotheses, preregistered
before running:

- **H1 (mechanism)** — the SAC dual holds the policy entropy at the
  target Ĥ = −dim(A)/2 = −0.5, which floors σ at the corresponding
  equilibrium width (≈ 0.15 for a 1-D tanh-Gaussian) instead of letting
  it decay/collapse.
- **H2 (the point of the experiment)** — with exploration kept alive,
  the entropy arm beats the plain global σ's best eval of −274.0 at the
  same budget.

H1 confirmed, precisely. H2 refuted: neither entropy arm comes within
34 points of the plain baseline.

## Method

### The switch

`SHACConfig.max_entropy = True` (CLI: `--max-entropy`) augments three
places, following SAPO's equations:

1. **Actor window return** (their Eq. 18-19): every window step's
   segment return carries `α·ĥ_t` in addition to the reward, with the
   same segmented-γ accumulation as the reward (the entropy term is a
   per-step scalar exactly like r_t, so it rides the identical
   γ/done machinery). The actor loss stays SHAC's FOBG window loss —
   the entropy enters through the same differentiable-simulation path.
2. **Critic TD-(λ) targets** (their Eq. 20): the recursion runs on
   `r + α·ĥ` — soft values. The critic then estimates the
   entropy-augmented return the actor optimizes.
3. **The α dual** (SAC Eq. 21, their Eq. 21): after each actor update,
   `α ← clamp₀₊(α − lr·(h − Ĥ))` on the window's mean raw entropy:
   below-target entropy raises α, above-target lowers it.

Design choices, all SAPO's unless noted:

- **Target Ĥ = −dim(A)/2 = −0.5** (SAC convention via Ball et al.
  2023; SAPO's choice).
- **Ĥ-normalized entropy in the losses** (their design choice II):
   ĥ = (h + |Ĥ|)/(2|Ĥ|) ≈ [0, 1], robust across reward scales. For
   CartPole's 1-D action 2|Ĥ| = 1, so ĥ = h + 0.5 — a constant shift
   that changes no actor gradient, only the (consistent) scale of the
   soft values. The dual sees the RAW h.
- **α init 1.0, entropy_lr 5e-3** (their Table 4), α clamped ≥ 0 so
   the bonus can never become a penalty.
- **Pre-dual α in the actor loss, post-dual α in the critic targets**
   (their pseudocode ordering).
- **SHAC's single critic + Polyak target kept** — this is SAPO-lite:
   their double-critic-without-targets (design choice IV) and
   SiLU/AdamW/clip changes (V) are NOT included, isolating the entropy
   objective itself.
- Works with BOTH σ parameterizations — the two experiment flags are
   orthogonal; `--state-std --max-entropy` is the faithful CartPole
   SAPO arm.
- Checkpoints load across the flag (α is stored as a separate
   `entropy_alpha` field, restored by max-entropy trainers and ignored
   by plain ones); the CSV grows `entropy_alpha` (the pre-dual α that
   entered that episode's actor loss) and `policy_entropy` (the
   window-mean raw h) — always present, empty cells when disabled, so
   the schema is identical across all arms.

### The entropy estimator

`h = log σ + ½log(2πe) + log(1 − tanh²(u) + ε)` at the reparameterized
sample u — SAC's −log π with the zero-mean `½(ε²−1)` term replaced by
its expectation (unbiased, lower variance). Differentiable in μ (the
tanh correction pushes away from action saturation) and in σ (log σ)
inside the window's computation graph. **The MC pin in test 1 caught a
real sign error here during development**: the change of variables
gives log π(a) = log N(u) − log(1−tanh²(u)), so the correction enters
the entropy with a PLUS sign (squashing *reduces* entropy — a
saturating action is low-entropy). The wrong-sign version overstated h
by 2·E[−log(1−tanh²(u))] (≈ 0.5 nats at the runs' initial σ ≈ 0.37)
and would have reported more entropy than the policy actually had.

### Correctness guarantee

`tests/test_max_entropy.py` (5 tests, direct-run like the rest of the
suite) pins: the estimator against a 200k-sample Monte-Carlo of the
true squashed-Gaussian entropy (and the below-Gaussian, tiny-σ, and
saturation limits); the dual's direction (init entropy above target →
α decreases from 1.0; forced tiny σ → α increases; the ≥ 0 clamp) and
that the logged `entropy_alpha` is the pre-dual value; the actor-loss
wiring (same-RNG-state plain vs entropy windows produce bit-identical
rollouts, and the loss gap equals the accumulated entropy contribution
exactly, with the entropy term adding direct gradient onto σ); the
critic wiring (soft targets run on `r + α·ĥ` with the post-dual α,
spied via the `td_lambda_values` call); and the train-loop integration
(identical CSV schema across arms, α schedule in the log, checkpoint
α round-trip, and the `--state-std --max-entropy` coexistence).

### The arms

| arm | config | episodes | train env steps | wall time |
|---|---|---|---|---|
| **global** | SHAC baseline (paper settings) | 500 | 1,024,000 | 86 s |
| **state** | `--state-std` | 500 | 1,024,000 | 87 s |
| **global+ent** | `--max-entropy` | 500 | 1,024,000 | 88 s |
| **state+ent** | `--state-std --max-entropy` | 500 | 1,024,000 | 90 s |

The first two arms are the STATE_STD.md runs reused as-is; the last
two are new. Same seed (0), same hyperparameters, run sequentially on
the same machine. Single-seed first-look protocol, like STATE_STD.md
(not the 5-seed protocol of the [critic ablation](ABLATION.md)); the
entropy arms' gap to the baseline (34-40 points) is smaller than the
ablation's seed scatter, so per-seed claims would need the fuller
protocol — the sign of the result is what we preregistered against.

## Results

### Returns

| | global | state | global+ent | state+ent |
|---|---|---|---|---|
| best eval return | **−274.0** (ep 500) | −311.8 (ep 490) | −314.3 (ep 480) | −308.6 (ep 490) |
| final eval return | −274.0 | −354.4 | −328.1 | −348.5 |
| mean of last 5 evals | **−322.7** | −350.0 | −368.7 | −349.2 |
| first eval > −400 | ep 110 | ep 360 | ep 240 | ep 210 |
| evals ahead of plain global | — | — | 19/50 | 13/50 |

Reading the curves (figure panel a): the entropy arms are competitive
early — global+ent is ahead of the plain global at 19 of 50 evals —
but the plain global's steadily-annealing σ grinds past them after
~episode 300 and finishes 34-40 points clear. The entropy arms' bests
all cluster at episodes 480-500: every arm was still improving at the
budget end, so the comparison is "who extracts the most from 1.02M
steps", and the answer is the plain objective.

### The dual works (H1): α, H, and the σ floor

| | global+ent | state+ent |
|---|---|---|
| α at ep 50 / 250 / 500 | 0.834 / 0.392 / **0.243** | 0.890 / 0.402 / **0.228** |
| final window entropy h | **−0.46** | **−0.58** |
| final policy σ | 0.163 | 0.142 |
| final train return (last-10-ep mean, stochastic) | −421.5 | −386.8 |

The dual behaves exactly as SAC's theory says: the init policy's
entropy (window-mean +0.10 / +0.16 at episode 1) is well above the
−0.5 target, so α decays monotonically and the entropy is pulled down
onto the target (−0.46 / −0.58 final; figure panel c). And the σ
consequence is the headline
mechanism result: **h = −0.5 sits at σ ≈ 0.15 for a 1-D
tanh-Gaussian, and the dual pins σ exactly there** — global+ent ends
at 0.163, state+ent at 0.142 (figure panel b). No annealing, no
collapse: the state mode's 0.0006 death is fully rescued to 0.142, and
the global mode's slow 0.372 → 0.058 decay is replaced by a flat floor.

### Exploration kept alive ≠ better at this budget

The entropy arms reach −400 much earlier than their plain
counterparts (state: ep 360 → 210; global: ep 110 → 240 — the latter
is the one place the plain arm is *faster*), and the state+ent arm
does beat its plain state sibling (−308.6 vs −311.8): with the entropy
keeping σ(s) alive, the state mode finally gets modest value from its
per-state adaptivity — the pattern SAPO's ablations predict. But 3
points is noise-adjacent, and neither entropy arm touches the plain
global's −274.0.

### Gradient-estimator variance

| Var[∇̂] at ep 25 / 500 | global | state | global+ent | state+ent |
|---|---|---|---|---|
| | 51 / 3.3 | 974 / 43.7 | 83 / 35.0 | 152 / 76.9 |

The entropy arms sit between the plain arms — the σ(s) head's
pure-noise ε-path still dominates the state+ent arm's early variance,
while the entropy's direct log-σ gradient adds a modest amount to the
global arm. Not the deciding factor here.

### Verdict

At the paper budget on CartPole: **keep the official objective.** The
max-entropy machinery is correctly implemented (all five tests pin it),
does precisely what SAC's theory says, and rescues the σ(s) collapse —
but the plain global σ's naturally-annealing exploration is the better
budget fit for this task.

## Interpretation

Why does keeping exploration alive not pay here? Two mechanisms, both
visible in the numbers:

1. **The objective is genuinely different, and the eval measures only
   one side of it.** After the dual settles (α ≈ 0.23, ĥ ≈ 0.5), the
   actor maximizes return + ≈ 0.1/step of entropy bonus. Mid-swing
   (|r| ~ 5-15) that's a rounding error; balanced (|r| ~ 0.1-1) it is
   10-100% of the reward — so late in training the entropy arm is
   optimizing "stay a bit random" against "hold the pole", while the
   deterministic eval (μ only) grades only the pole. The final
   stochastic train returns make the same point from the other side:
   the entropy arms explore at σ ≈ 0.15 through the whole budget.
2. **This budget doesn't have an exploration problem left to solve.**
   The plain global reaches −400 by ep 110 and its σ anneals 0.372 →
   0.058 — an exploration schedule that happens to match the task's
   difficulty curve. The entropy floor replaces that schedule with a
   flat 0.15: more exploration exactly when the plain arm is starting
   to exploit. All four arms were still improving at ep 500, so the
   honest reading is not "entropy hurts the converged policy" but
   "entropy spends the 1.02M-step budget more slowly".

## Relation to SAPO

SAPO's wins are on tasks where SHAC's short-horizon training stalls in
local minima — contact-rich rigid-body locomotion (+6.8%) and the
deformable HandFlip (+172.7%, of which +69.7% from the non-entropy
design choices {III,IV,V} alone). Their critical ablation (b) shows
removing the entropy from the actor loss costs most of the gain *on
those tasks*. Nothing in our result contradicts that: CartPole
Swing-Up at the paper's 1.02M-step budget is the easy end of the
spectrum — the plain objective's remaining gap at ep 500 is a budget
gap, not a local-minimum gap, and entropy insurance that costs
~34 eval points buys nothing when there is no pathology to insure
against. What this experiment adds is the other side of SAPO's scope
boundary: at toy scale and budget, the SAC entropy term is a pure cost
— and their own rigid-body ablation points the same direction
({III,IV,V} without entropy: +0.5% on DFlex, i.e. also not where the
gain is).

## Reproducing

```bash
# the two new arms (sequential, same machine, ~90 s each); the plain
# arms are the STATE_STD.md runs (see that document's commands)
uv run python -m shac.train --logdir runs/std_ent_state_500ep --seed 0 \
    --episodes 500 --device cpu --threads 1 --state-std --max-entropy
uv run python -m shac.train --logdir runs/std_ent_global_500ep --seed 0 \
    --episodes 500 --device cpu --threads 1 --max-entropy

# the four-panel figure  -> figures/max_entropy_ab.png
uv run python scripts/plot_max_entropy.py

# per-run training figures
uv run python scripts/plot_training.py runs/std_ent_state_500ep
uv run python scripts/plot_training.py runs/std_ent_global_500ep

# the five correctness tests
uv run python tests/test_max_entropy.py
```

Artifacts (gitignored `runs/`, regenerable above):
`runs/std_ent_state_500ep/` and `runs/std_ent_global_500ep/` (log.csv,
checkpoints), the committed figure `figures/max_entropy_ab.png`.
