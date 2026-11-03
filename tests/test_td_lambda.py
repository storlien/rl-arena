"""
Correctness test for the td-(lambda) value estimation (paper Eq. 7).

The trainer uses an efficient reverse recursion (see
shac.algorithms.shac.td_lambda_values, which also matches the official
implementation). Here we validate it against a NAIVE, literal implementation
of the paper's Eq. 7:

    V-tilde(s_t) = (1-lambda) * sum_{k=1}^{n-1} lambda^{k-1} * G^k_t
                   + lambda^{n-1} * G^n_t

with the segment generalization for episode resets:

  * n = number of rewards remaining in the env's current task-episode
    segment (bounded by the first done at or after t; the window's last
    step acts as a done for every env).
  * G^k_t = (sum_{j<k} gamma^j * r_{t+j}) + gamma^k * V(s_{t+k}), where the
    bootstrap for the segment's final k uses the TERMINAL value supplied in
    next_value_buf (0 for early termination, V(pre-reset state) for a time
    limit, V(s_h) at the window end).

Randomized over rewards, values, and done patterns (including multiple
resets per window and resets on the final step).

Run directly:  uv run python tests/test_td_lambda.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from shac.algorithms.shac import td_lambda_values


def naive_td_lambda(rew, next_v, done_mask, gamma, lam):
    """Literal, slow, obviously-correct Eq. 7 with segment truncation.

    rew, next_v, done_mask: [h, N].  done_mask[h-1] must be all ones."""
    h, n = rew.shape
    out = torch.zeros_like(rew)
    for e in range(n):                      # each env independently
        # segment end for each t: first index >= t with done_mask == 1
        for t in range(h):
            end = t
            while done_mask[end, e] < 0.5:
                end += 1
            n_steps = end - t + 1          # rewards available from s_t

            def g(k):                       # k-step return from s_t
                ret = 0.0
                for j in range(k):
                    ret += gamma ** j * rew[t + j, e]
                return ret + gamma ** k * next_v[t + k - 1, e].item()

            if n_steps == 1:
                out[t, e] = g(1)            # only the final return exists
            else:
                acc = 0.0
                for k in range(1, n_steps):
                    acc += lam ** (k - 1) * g(k)
                out[t, e] = (1 - lam) * acc + lam ** (n_steps - 1) * g(n_steps)
    return out


def test_matches_naive():
    torch.manual_seed(0)
    h, N, trials = 12, 5, 30
    for trial in range(trials):
        rew = torch.randn(h, N, dtype=torch.float64)
        next_v = torch.randn(h, N, dtype=torch.float64)
        done_mask = (torch.rand(h, N, dtype=torch.float64) < 0.15).float()
        done_mask[h - 1] = 1.0              # window end: bootstrap boundary
        if trial == 0:
            done_mask.zero_(); done_mask[h - 1] = 1.0   # no resets at all
        if trial == 1:
            done_mask[3] = 1.0              # all envs reset mid-window

        fast = td_lambda_values(rew, next_v, done_mask, gamma=0.99, lam=0.95)
        slow = naive_td_lambda(rew, next_v, done_mask, gamma=0.99, lam=0.95)
        err = (fast - slow).abs().max().item()
        assert err < 1e-10, f"trial {trial}: mismatch {err:.2e}"
    print(f"  vectorized td-(lambda) == naive Eq. 7 over {trials} "
          f"random cases (max err {err:.2e})")


def test_lambda_one_is_monte_carlo():
    """lambda -> 1 must reduce to the plain n-step (Monte-Carlo) return to
    the segment end -- a sanity check of the weighting scheme.

    NOTE: lambda = 1 EXACTLY is a degenerate case of the (official)
    recursion, which contains the ratio (1-lam_pow)/(1-lam) = 0/0 there;
    the paper always uses lambda < 1 (0.95), so we probe the limit instead.
    """
    torch.manual_seed(1)
    h, N = 8, 3
    rew = torch.randn(h, N, dtype=torch.float64)
    next_v = torch.randn(h, N, dtype=torch.float64)
    done_mask = torch.zeros(h, N, dtype=torch.float64)
    done_mask[h - 1] = 1.0

    lam = 1.0 - 1e-6
    vals = td_lambda_values(rew, next_v, done_mask, gamma=0.9, lam=lam)
    for e in range(N):
        expected = 0.0
        for j in range(h):                  # full MC return to window end
            expected += 0.9 ** j * rew[j, e]
        expected += 0.9 ** h * next_v[h - 1, e].item()
        assert abs(vals[0, e].item() - expected) < 5e-3, \
            f"env {e}: {vals[0, e].item()} vs MC {expected}"
    print("  lambda -> 1 reduces to the Monte-Carlo return (OK)")


if __name__ == "__main__":
    test_matches_naive()
    test_lambda_one_is_monte_carlo()
    print("ALL TD-LAMBDA TESTS PASSED")
