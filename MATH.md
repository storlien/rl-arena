# Formulas: SHAC and the no-critic ablation

Every formula implemented in this repo, in training-loop order. Equation
numbers refer to the SHAC paper (Rypgaard et al., "Short-Horizon Actor-
Critic"); "official" = the authors' released code (NVlabs/DiffRL). Values
are the ones actually used here (paper Tables 1/4/5, cross-checked against
the official `cartpole_swing_up.yaml`). File references point to this
implementation.

Notation: `N` = parallel envs, `h` = short horizon (window), `H` = task
horizon, `M` = learning episodes, `gamma` = discount, `lambda` = TD(λ).
All display math uses `$$` blocks with the delimiters on their own lines.

---

## 0. The MDP (paper Appendix A.3.1)

**Observation** (paper Table 6), with `q = (x, theta)`, theta measured
**from upright**:

$$
o_t = \big[x_t,\ \dot{x}_t,\ \sin(\theta_t),\ \cos(\theta_t),\ \dot{\theta}_t\big] \in \mathbb{R}^5
$$

**Applied action** — raw network output squashed and scaled to the effort
limit (official code: `env.step(torch.tanh(actions))`):

$$
u_t = 1000 \tanh(a_t)\ \mathrm{N}
$$

**Reward** (paper Eq. 13), angle wrapped to (-π, π] first so a spinning
pole is not rewarded for accumulated angle:

$$
\tilde{\theta} = \operatorname{atan2}(\sin(\theta),\ \cos(\theta))
$$

$$
r_t = -\tilde{\theta}_t^2 - 0.1\,\dot{\theta}_t^2 - 0.05\,x_t^2 - 0.1\,\dot{x}_t^2
$$

(always ≤ 0; a perfectly balanced pole earns ≈ 0 per step.)

**Dynamics** — Euler–Lagrange equations of the rigid pole, solved in
closed form. Parameters (official URDF): cart M = 1 kg, pole m = 10 kg,
COM distance l = 0.5 m, `I_com` = 1 kg·m², g = 9.81:

$$
I = I_{\mathrm{com}} + m\,l^2 = 3.5\ \mathrm{kg \cdot m^2}
\quad \text{(parallel axis)}
$$

$$
\begin{bmatrix} M+m & m\,l\cos(\theta) \\ m\,l\cos(\theta) & I \end{bmatrix}
\begin{bmatrix} \ddot{x} \\ \ddot{\theta} \end{bmatrix}
=
\begin{bmatrix} u + f_{\mathrm{limit}} + m\,l\sin(\theta)\,\dot{\theta}^2 \\ m\,g\,l\sin(\theta) \end{bmatrix}
$$

With `s = sin(theta)`, `c = cos(theta)` and:

$$
\Delta = (M+m)\,I - (m\,l\cos(\theta))^2 > 0
$$

(positive-definite mass matrix, so a plain division is safe and smooth):

$$
\ddot{x} = \frac{I\,\big(u + f_{\mathrm{limit}} + m\,l\,s\,\dot{\theta}^2\big) - m\,l\,c \cdot m\,g\,l\,s}{\Delta}
$$

$$
\ddot{\theta} = \frac{(M+m)\,m\,g\,l\,s - m\,l\,c\,\big(u + f_{\mathrm{limit}} + m\,l\,s\,\dot{\theta}^2\big)}{\Delta}
$$

**Soft joint limit** on the cart (paper Eq. 4; engaged only outside
[-4, 4] m, k = 100, `c_lim` = 1):

$$
f_{\mathrm{limit}} = 100 \cdot \mathrm{viol}(x) - 1.0\,\dot{x}\cdot\mathbf{1}[\mathrm{viol}(x) \neq 0]
$$

**Integration** — semi-implicit (symplectic) Euler; one policy step of
dt = 1/60 s is 4 substeps of dt/4 = 1/240 s, action held constant:

$$
\dot{q} \leftarrow \dot{q} + \tfrac{dt}{4}\,\ddot{q},
\qquad
q \leftarrow q + \tfrac{dt}{4}\,\dot{q}
\qquad (\times 4 \text{ per action})
$$

**Initial state** (`U` uniform on [0, 1]; pole starts hanging, θ₀ = -π):

$$
q_0 = (0,\ -\pi) + \pi\,(U - 0.5),
\qquad
\dot{q}_0 = (0,\ 0) + 0.5\,(U - 0.5)
$$

Episodes end **only at the time limit** H = 240 steps (no early
termination). Implementation: `shac/envs/cartpole_swingup.py`.

## 1. Networks and preprocessing

**Actor** (paper A.2.2 + official code) — Gaussian with a *global
learnable* standard deviation, sampled with the reparameterization trick:

$$
\mu_\phi(\hat{o}) = \mathrm{MLP}_{64,64}(\hat{o}),
\qquad
a_t = \mu_\phi(\hat{o}_t) + \sigma\,\epsilon_t,
\qquad
\epsilon_t \sim \mathcal{N}(0, 1)
$$

$$
\sigma = e^{\mathrm{logstd}},
\qquad
\mathrm{logstd}_0 = -1
\;\Rightarrow\;
\sigma_0 \approx 0.37
$$

MLP: `Linear -> ELU -> LayerNorm` per hidden layer, orthogonal init
(gain √2). No entropy bonus — sigma is learned only through the
reparameterized pathwise gradient. Evaluation uses the mean action
(implementation: `shac/models/networks.py`):

$$
u_{\mathrm{eval}} = 1000 \tanh\big(\mu_\phi(\hat{o})\big)
$$

**Critic**: `V(ô)` = `MLP[64,64](ô) -> scalar`, plus a frozen target copy
`V'` — its parameters never receive gradients, but gradients flow through
its *input* (`dV'/ds`, needed for the terminal adjoint).

**Observation normalization** (paper Sec. 3.3) — running statistics merged
per batch with the Chan et al. parallel-variance algorithm, **frozen for
the duration of one learning episode** (snapshot), updated in the
background (implementation: `shac/utils/running_stats.py`):

$$
\hat{o} = \frac{o - \mu_{\mathrm{run}}}{\sqrt{\sigma^2_{\mathrm{run}} + 10^{-5}}}
$$

$$
\delta = \bar{o} - \mu,
\qquad
n_{\mathrm{tot}} = n + n_b
$$

$$
\mu \leftarrow \mu + \delta\,\frac{n_b}{n_{\mathrm{tot}}},
\qquad
\sigma^2 \leftarrow \frac{\sigma^2\,n + s_b^2\,n_b + \delta^2\,n\,n_b / n_{\mathrm{tot}}}{n_{\mathrm{tot}}}
$$

## 2. Stage 1 — rollout

`N` = 64 envs roll `h` steps; the autograd graph is kept **inside** the
window and cut **between** learning episodes (`q <- detach(q)`,
`initialize_trajectory`). Windows continue physically from where the
previous learning episode ended.

## 3. Stage 2 — actor loss (paper Eq. 5, segmented)

Per-env running accumulators for the current task-episode segment, with
`a` the window-relative segment start. The discount power **resets to
gamma^0 at every done** (official handling of task termination). Code
names: `rew_acc` / `gamma_pow`.

$$
S_e(t) = \sum_{\tau=a}^{t} \gamma^{\tau-a}\, r_\tau,
\qquad
D_e(t) = \gamma^{t-a}
$$

Each env contributes its segment when it finishes (done inside the
window); all remaining envs contribute at the window end:

$$
C_e = S_e(t) + \gamma\, D_e(t)\, \tilde{v}(s_{t+1})
$$

with the terminal-value override:

$$
\tilde{v}(s_{t+1}) =
\begin{cases}
V'(s_{t+1}) & \text{window end } (t = h-1,\ \text{all envs}) \\
V'(s_{\text{pre-reset}}) & \text{done at time limit} \\
0 & \text{done by early termination (never in CartPole)}
\end{cases}
$$

(the time-limit case bootstraps the *pre-reset* state — the task would
continue, only the counter resets). Actor loss, then **one Adam step**
(betas 0.7, 0.95):

$$
L_{\mathrm{actor}} = -\frac{1}{N h} \sum_{e=1}^{N} \sum_{\mathrm{segments}} C_e
$$

With no dones and `a = 0` this is exactly paper Eq. 5:

$$
C = \sum_{t=0}^{h-1} \gamma^t\, r_t + \gamma^h\, V'(s_h)
$$

Backprop through the differentiable simulator gives the analytic pathwise
gradient; the terminal adjoint arises automatically because `V'` is
applied to the graph-connected terminal state (paper Eq. 9). Gradient
clipping (both networks), with a crash guard on non-finite norms:

$$
\hat{g} = g \cdot \min\Big(1,\ \frac{1.0}{\lVert g \rVert_2}\Big)
$$

## 4. Stage 3 — critic loss (paper Eq. 6–7)

For every visited state, the k-step return (`n` = rewards left in the
segment) and the TD(λ) mixture (Eq. 7), `lambda` = 0.95:

$$
G^k_t = \sum_{j=0}^{k-1} \gamma^j\, r_{t+j} + \gamma^k\, \tilde{v}(s_{t+k})
$$

$$
\tilde{V}(s_t) = (1-\lambda) \sum_{k=1}^{n-1} \lambda^{k-1}\, G^k_t + \lambda^{n-1}\, G^n_t
$$

The implementation uses the official reverse recursion, processing
`i = h-1 ... 0` with done mask `dm_i` (forced to 1 at the window end,
which acts as a bootstrap boundary for every env):

$$
\Lambda_i \leftarrow \lambda\,\Lambda_{i+1}\,(1 - \mathrm{dm}_i) + \mathrm{dm}_i
$$

$$
A_i \leftarrow (1 - \mathrm{dm}_i)\Big( \lambda\,\gamma\, A_{i+1} + \gamma\, \tilde{v}(s_{i+1}) + \frac{1 - \Lambda_i}{1 - \lambda}\, r_i \Big)
$$

$$
B_i \leftarrow \gamma\big( \tilde{v}(s_{i+1})\, \mathrm{dm}_i + B_{i+1}\,(1 - \mathrm{dm}_i) \big) + r_i
$$

$$
\tilde{V}(s_i) = (1-\lambda)\, A_i + \Lambda_i\, B_i
$$

with `B_i` = the full n-step return and the `A` part = the k < n mixture
(verified against the naive k-loop in `tests/test_td_lambda.py`).
Regression loss (Eq. 6), fitted with **16 iterations × 4 contiguous
time-major minibatches = 64 Adam steps** per learning episode, same clip,
plus the official `nan_to_num` sanitization of non-finite critic
gradients:

$$
L_{\mathrm{critic}}(\varphi) = \frac{1}{h N} \sum_{t,e} \big( V_\varphi(\hat{s}_{t,e}) - \tilde{V}(s_{t,e}) \big)^2
$$

## 5. Stage 4 — target update (paper Algorithm 1)

$$
V' \leftarrow \alpha\, V' + (1 - \alpha)\, V,
\qquad
\alpha = 0.2
$$

## 6. Learning-rate schedule (official)

Linear decay over the `M` learning episodes, per network; actor 1e-2,
critic 1e-3, both toward 1e-5:

$$
\mathrm{lr}(m) = \mathrm{lr}_0 + \big(\mathrm{lr}_{\mathrm{final}} - \mathrm{lr}_0\big) \cdot \min\Big(1,\ \frac{m}{M}\Big)
$$

## 7. The no-critic ablation (`--no-critic`)

One substitution, `tilde{v} ≡ 0` (bit-identical to a critic whose outputs
are zeroed — verified in `tests/test_no_critic.py`); stages 3–4 are
skipped entirely:

$$
L_{\mathrm{nc}} = -\frac{1}{N h} \sum_{e=1}^{N} \sum_{\mathrm{segments}} \sum_{\tau=a}^{t} \gamma^{\tau-a}\, r_\tau
$$

**Myopia view** — what lies beyond the window edge enters the update
objective *as an estimated value* in full SHAC (`+ gamma^h * V'(s_h)`),
*not at all* in the no-critic arms with `h < H`, and *does not exist* for
arm C (`h = H = 240`, every window is exactly one episode, `a = 0`
always) — its objective **is** the full discounted episode return:

$$
L_{C} = -\frac{1}{N} \sum_{e=1}^{N} \sum_{t=0}^{239} \gamma^t\, r_t
$$

Zero truncation gap — and the gradient is the full 240-step Jacobian
product through the unstable dynamics (see ABLATION.md for why that
explodes).

## 8. Evaluation

Raw, undiscounted return of one full episode under the deterministic mean
policy, on a separate environment:

$$
\mathrm{return} = \sum_{t=0}^{239} r_t
$$

## 9. Values as run

| Symbol | Meaning | Value |
|---|---|---|
| N | parallel envs | 64 |
| h | short horizon | 32 (A, B) / 64 (D) / 240 (C) |
| H | task horizon | 240 steps (4 s) |
| M | learning episodes | 500 (A, B) / 67 (C) / 250 (D) |
| gamma | discount | 0.99 |
| lambda | TD(λ) | 0.95 |
| lr | actor / critic | 1e-2 / 1e-3 → 1e-5 linear |
| betas | Adam | 0.7, 0.95 |
| — | grad-norm clip | 1.0 (both networks) |
| alpha | Polyak | 0.2 |
| — | critic updates / episode | 16 iter × 4 batches |
| — | MLPs | [64, 64], ELU + LayerNorm |
| sigma_0 | action std init | e^-1 ≈ 0.37 |

All arms consume ≈ 1.02M env steps (`M * N * h`).
