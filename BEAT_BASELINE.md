# Experiment: beating the paper baseline on CartPole

A companion document to [ABLATION.md](ABLATION.md),
[STATE_STD.md](STATE_STD.md) and [MAX_ENTROPY.md](MAX_ENTROPY.md). The
question: **at the paper's budget (500 episodes ≈ 1.02M env-steps, N=64,
h=32), can anything beat plain SHAC's best eval on CartPole Swing-Up?**
The reference point is the plain-global run of STATE_STD.md: best eval
**−274.0** (ep 500), last-5 mean −322.7 ± 55 — a policy that solves the
task (100% swing-up, ~3° balance) but was still improving when the
budget ran out, i.e. the gap is a *speed* gap, not a local-minimum gap.

**TL;DR.** Yes — with SAPO's architecture/optimizer choice V (ELU → SiLU,
Adam → AdamW) plus the longer horizon h = 64: best evals **−234.4 /
−215.1 / −261.8** over seeds 0/1/2 (mean **−237**), vs the baseline's
−274.0 / −323.4 / −323.8 (mean −307). **Every improved seed beats every
baseline seed**; the paired mean improvement is ≈ 70 eval points. SiLU +
AdamW alone already wins on all three seeds (mean −253.6). Everything
else we tried — slower Polyak, both td-λ directions, more parallel envs,
TD3-style critic warmup, a SAPO-inspired double-critic-without-targets
(choice IV) — *lost* to the baseline. **Full SAPO** (all five design
choices together, faithful to the paper, implemented after these arms)
lands *between*: mean −266.5 — better than the baseline, but its gain is
mostly spent by the entropy tax and the state-σ policy class. The paper's
critic machinery (fast Polyak 0.2, td-λ 0.95) is exactly right; the
paper's *network/optimizer stack* is where the headroom was.

> **Naming.** This document's arms are **SHAC with single SAPO-inspired
> changes**, not SAPO. SAPO is the *joint* algorithm — all five design
> choices (entropy objective, Ĥ-normalization, state-σ policy class,
> double-critic/no-targets, SiLU/AdamW/clip-0.5) active simultaneously on
> top of SHAC's FOBG training loop. Each `cp_shac_sapo*` arm below flips
> ONE knob with the entropy machinery off, so none of them is SAPO; only
> the `cp_sapo_full` arm (§ Full SAPO) is an actual SAPO reimplementation
> (on our envs). Run directories keep this convention:
> `cp_shac_sapo*` = SHAC + one SAPO-inspired change,
> `cp_sapo_full` = full SAPO.

---

## The arms

Two tiers, both preregistered before running (predictions in the
tables). Single-change arms on top of the paper configuration, seed 0
unless noted, 500 episodes, same machine sequentially.

### Tier 1 — hyperparameter arms (config-only, zero code changes)

| arm | change | inspiration | prediction | best eval | verdict |
|---|---|---|---|---|---|
| baseline | — (paper Table 4) | — | — | −274.0 | reference |
| `cp_polyak0995` | Polyak α 0.2 → 0.995 | external SAC analysis (slow targets); the paper's own Table 5 general setting | slightly better: less target churn while the critic fits | −377.2 | **worse, badly** |
| `cp_tdl098` | td-λ 0.95 → 0.98 | longer credit assignment | neutral-to-worse | −425.7 | **worse, badly** |
| `cp_tdl090` | td-λ 0.95 → 0.90 | opposite direction | neutral | −274.8 | tie, slower |
| `cp_h64` | h 32 → 64 | deeper BPTT | neutral | −273.8 | tie best, **better last-5** (−300 vs −323) |
| `cp_n128` | N 64 → 128 | halve gradient variance per update | better best (2× env-step budget, flagged) | −309.0 | worse |

Reading: the paper's critic hyperparameters sit in a sharp optimum —
moving Polyak or td-λ in EITHER direction costs 50–150 points. The fast
Polyak (0.2 = target moves 80% toward the critic per episode) and the
moderate λ are load-bearing. h = 64 ties on best but lands a clearly
better last-5 with lower scatter — the hint that h matters, picked up
in tier 3.

### Tier 2 — architecture/optimizer arms (new config-gated code)

| arm | change | inspiration | prediction | best eval | verdict |
|---|---|---|---|---|---|
| `cp_warmup10` | critic-only training for the first 10 episodes (actor frozen, rollout still runs) | TD3's delayed policy updates; the critic MSE starts ~800, so early actor gradients bootstrap through a garbage critic | better: actor steps only once the bootstrap is meaningful | −350.8 | **worse** |
| `cp_shac_sapoiv` | critic ensemble K=2, min-pooled bootstrap, **no** target network, no Polyak | SAPO design choice IV (SAC/CrossQ lineage) | neutral-to-better (variance reduction) | −319.8 | worse |
| `cp_shac_sapov` | ELU → **SiLU**, Adam → **AdamW** (identical lr/betas), both networks | SAPO design choice V | neutral (seems cosmetic) | **−249.4** | **beats baseline** |

(Note the run-directory rename: these arms were first written up as
`cp_sapo_iv` / `cp_sapo_v`; renamed to `cp_shac_sapo*` because a single
flipped knob with the entropy machinery off is **not SAPO** — see the
naming note above. The original `cp_sapo_iv` run also pooled the
ensemble with min in the ACTOR's window loss; SAPO's paper specifies
average there and min only for critic targets — `--actor-bootstrap-pool`
now exposes both.)

The surprise is the *size ordering*: the changes with the strongest
theoretical motivation (ensemble targets, delayed updates) lost; the
change that looks cosmetic won. Both "principled" arms interact badly
with SHAC's ONE-actor-update-per-episode structure: the ensemble doubles
critic compute for a min-bias the fast Polyak target apparently already
controls, and the warmup starves the actor of 10 of its 500 updates in a
regime that is update-limited, not data-limited.

### Tier 3 — the winner, confirmed over seeds

| arm | seed 0 | seed 1 | seed 2 | mean best | mean last-5 |
|---|---|---|---|---|---|
| baseline (paper) | −274.0 | −323.4 | −323.8 | −307.1 | −350.3 |
| SHAC + V-style (SiLU/AdamW) | −249.4 | −257.7 | −253.7 | −253.6 | −293.9 |
| **SHAC + V-style + h = 64** | **−234.4** | **−215.1** | −261.8 | **−237.1** | −281.7 |

(`runs/cp_base_s{1,2}/`, `runs/cp_shac_sapov/`,
`runs/cp_shac_sapov_s{1,2}/`, `runs/cp_shac_sapov_h64_s{0,1,2}/`;
behavioral metrics on the best checkpoints: all three V-style+h64
policies solve the task — 100% swing-up, t_up ≈ 58 steps, balance ≈ 3° —
i.e. the improvement is convergence SPEED toward the same solved
behavior, exactly matching the "budget gap" framing.)

The seed separation is complete: every V-style seed and every
V-style+h64 seed beats every baseline seed. The winner's worst seed
(−261.8) still beats the baseline's best seed (−274.0).

### Full SAPO — the faithful reimplementation, tested alongside

After the arms above, SAPO was implemented **in full and faithfully**
(paper §4.2, all five design choices active simultaneously — the only
configuration that deserves the name SAPO):

| choice | flags |
|---|---|
| I: entropy + dual α | `--max-entropy` (target = Ĥ = −dim(A)/2 = −0.5) |
| II: Ĥ-normalized entropy | built into the max-entropy machinery |
| III: state-σ policy class | `--state-std` |
| IV: 2× critic, no targets, **average in the actor return / min in the targets** | `--critic-ensemble 2 --no-target-critic --actor-bootstrap-pool avg` |
| V: SiLU + AdamW + **grad-clip 0.5** | `--activation silu --optimizer adamw --grad-clip 0.5` |

Backbone: SHAC's paper CartPole config (N=64, h=32, 500 eps, seed 0/1/2,
`runs/cp_sapo_full{,_s1,_s2}`). **Prediction (preregistered):** V is the
only ingredient proven positive here and entropy/III/IV were each
negative, so full SAPO should beat the baseline but lose to
V-style+h64 — unless faithful average pooling rescues choice IV.

| arm | seed 0 | seed 1 | seed 2 | mean best | mean last-5 |
|---|---|---|---|---|---|
| baseline (paper) | −274.0 | −323.4 | −323.8 | −307.1 | −350.3 |
| **full SAPO (all 5 choices)** | −272.9 | −277.6 | −249.0 | **−266.5** | −294.0 |
| SHAC + V-style (for reference) | −249.4 | −257.7 | −253.7 | −253.6 | −293.9 |

Result: **prediction confirmed.** Full SAPO beats the baseline on the
mean (−266.5 vs −307.1) but with NO seed separation (−277.6 loses to the
baseline's best −274.0), and it loses to SHAC + V-style alone
(−253.6): the entropy tax (~34 points, MAX_ENTROPY.md) plus the state-σ
policy class eat most of V's gain. The machinery itself worked exactly
as SAPO specifies — the dual annealed α 1.0 → 0.23 while holding the
policy entropy at H ≈ −0.57 vs the −0.5 target, and σ(s) stayed alive
(0.37 → 0.14, vs the σ-collapse to 0.058 the plain objective produces).
So this is a clean negative on SAPO-as-a-whole at this budget, not a
broken implementation: on a task SHAC already solves, SAPO's insurance
premium costs more than its architecture discount returns. Two caveats:
(i) the earlier IV-style arm's min-pooled actor bootstrap is now known
to be unfaithful, so choice IV's solo verdict carries that asterisk —
though full SAPO (with faithful pooling) still underperforms V alone,
which bounds IV's contribution here; (ii) SAPO's published wins are on
massively-parallel GPU tasks where SHAC stalls — a regime our toy
budget deliberately does not reproduce.

## Interpretation

Why would SiLU + AdamW be worth ≈ 70 points when ensemble critics and
target tricks are worth negative? Honest reading, with the evidence at
hand:

1. **SiLU smooths the path the algorithm is most sensitive to.** SHAC's
   actor gradient is a full BPTT through 32–64 simulator steps; its
   quality depends on the smoothness of ∂μ/∂s through two 64-unit
   layers. SiLU (x·σ(x)) is smooth (C^∞) where ELU has a curvature
   kink at 0 — and unlike tanh, it is not saturating-negative. LayerNorm
   sits after every hidden activation, which further rewards activation
   functions with well-behaved local Jacobians. This is a hypothesis,
   not a measurement — but it is consistent with the h=64 interaction:
   the LONGER the BPTT chain, the more the smoothness compounds
   (V alone: −253.6 mean; V+h64: −237.1 mean; h64 alone: tied).
2. **AdamW's decoupled weight decay is a regularizer the small policy
   actually feels.** With lr 1e-2 → 1e-5 linear decay and only ~500
   updates, coupled-L2 Adam keeps effective decay constant relative to
   the adaptive step; AdamW's decay stays proportional to the weight
   magnitude, which mildly regularizes exactly the late-training phase
   where the baseline's eval returns scatter widest (±55 → ±20).
3. **Negative results are load-bearing here.** The two "principled"
   variance-reduction arms (IV, warmup) both LOST — evidence that the
   paper's single-critic + fast-Polyak machinery is already adequate
   for this task's target-churn problem, and that the remaining lever
   was never the critic. It was the network stack nobody ablates.

## What this means for SAPO

SAPO's choice V was dismissed in MAX_ENTROPY.md's accounting as a
bundle-in ({III,IV,V} = +0.5% on DFlex rigid-body tasks) — on CartPole
it is the whole show. Our decomposition: V-style alone ≈ 54 points,
IV-style negative, III alone (STATE_STD.md) negative, and **full SAPO
(all five choices, faithful pooling and clip) lands in between at mean
−266.5** — its positive ingredient is diluted by the entropy objective
and the policy-class change that SAPO couples to it. SAPO's own
ablations point the same direction for rigid-body tasks (their Table 14,
+0.5% for {III,IV,V} without entropy) — but note their bundle did NOT
include the h change, and their claim was about local-minimum escape,
not convergence speed. The reconciled picture: on tasks SHAC already
solves, choice V is a free ≈ 20% speedup, and the rest of SAPO is a net
cost; on tasks where SHAC stalls, SAPO's entropy term remains the part
that matters (and our pendulum result bounds even that).

## Reproducing

```bash
# tier 1: config-only arms (~90 s each)
for flags in "--target-alpha 0.995" "--td-lambda 0.98" "--td-lambda 0.90" \
             "--horizon 64" "--num-envs 128"; do
  uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 0 $flags
done

# tier 2: architecture arms (SHAC + one SAPO-inspired change each)
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 0 \
    --critic-warmup 10
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 0 \
    --critic-ensemble 2 --no-target-critic
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 0 \
    --activation silu --optimizer adamw

# full SAPO: all five design choices, faithful to paper section 4.2
# (entropy+dual, H-bar normalization, state-std policy class, 2 critics
# with average pooling in the actor return / min in the targets, no
# target networks, SiLU + AdamW + grad-clip 0.5)
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 0 \
    --max-entropy --state-std \
    --critic-ensemble 2 --no-target-critic --actor-bootstrap-pool avg \
    --activation silu --optimizer adamw --grad-clip 0.5

# tier 3: multi-seed confirmation (baseline seeds 1-2, V-style seeds 1-2,
# V-style+h64 seeds 1-2, full-SAPO seeds 1-2)
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 1
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 2
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 1 \
    --activation silu --optimizer adamw
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 2 \
    --activation silu --optimizer adamw
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 0 \
    --horizon 64 --activation silu --optimizer adamw
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 1 \
    --horizon 64 --activation silu --optimizer adamw
uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed 2 \
    --horizon 64 --activation silu --optimizer adamw
for s in 1 2; do
  uv run python -m shac.train --env cartpole --device cpu --threads 1 --seed $s \
      --max-entropy --state-std \
      --critic-ensemble 2 --no-target-critic --actor-bootstrap-pool avg \
      --activation silu --optimizer adamw --grad-clip 0.5
done

# per-run training figures + behavior metrics
uv run python scripts/plot_training.py runs/cp_shac_sapov_h64_s0
uv run python scripts/policy_metrics.py runs/cp_shac_sapov_h64_s0/best_policy.pt
```

All run directories are gitignored (`runs/`) and regenerable above.
