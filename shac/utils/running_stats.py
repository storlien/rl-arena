"""
Running mean / variance statistics for observation normalization.

This mirrors the paper's observation normalization (Section 3.3):
    "we apply observation normalization as is common in RL algorithms, which
     normalizes the state observation by a running mean and standard
     deviation calculated from the state observations in previous learning
     episodes."

Implementation details follow the official code (utils/running_mean_std.py
in NVlabs/DiffRL), which uses the classic parallel-variance algorithm
(Chan et al.): merging a batch of statistics into the running estimates
without ever storing the raw history.

How SHAC uses this (important subtlety, matches the official code):
  * At the START of each learning episode we take a frozen snapshot of the
    current (mean, var).
  * All normalization inside that episode (actor input, critic input,
    target-critic input) uses the frozen snapshot, so the entire episode's
    computation graph is consistent.
  * The live statistics are still UPDATED with every observation seen during
    the episode, so the *next* episode uses fresher statistics.
"""

import torch


class RunningMeanStd:
    """Tracks the running mean and variance of a stream of vectors.

    Attributes:
        mean:  running mean, shape `shape`
        var:   running (population) variance, shape `shape`
        count: number of samples incorporated so far (float, starts at
               `epsilon` to avoid division-by-zero on the first updates)
    """

    def __init__(self, shape: tuple, device: str = "cpu", epsilon: float = 1e-4):
        self.mean = torch.zeros(shape, dtype=torch.float32, device=device)
        self.var = torch.ones(shape, dtype=torch.float32, device=device)
        self.count = epsilon
        self.device = device

    # ------------------------------------------------------------------ update
    @torch.no_grad()
    def update(self, batch: torch.Tensor) -> None:
        """Merge a batch of observations [..., shape] into the statistics."""
        batch_mean = batch.mean(dim=0)
        # unbiased=False -> population variance of the batch
        batch_var = batch.var(dim=0, unbiased=False)
        batch_count = batch.shape[0]

        # --- Chan et al. parallel merge ---
        delta = batch_mean - self.mean
        total = self.count + batch_count

        new_mean = self.mean + delta * (batch_count / total)
        m_a = self.var * self.count                 # sum of squared deviations, old
        m_b = batch_var * batch_count               # ... , batch
        m2 = m_a + m_b + delta.square() * self.count * batch_count / total
        new_var = m2 / total

        self.mean = new_mean
        self.var = new_var
        self.count = total

    # --------------------------------------------------------------- normalize
    def normalize(self, arr: torch.Tensor) -> torch.Tensor:
        """Return (arr - mean) / sqrt(var + eps).

        Kept differentiable: mean/var are treated as constants here (they
        are updated under no_grad and never require grad), so gradients flow
        through `arr` only -- exactly what we want inside the rollout graph.
        """
        return (arr - self.mean) / torch.sqrt(self.var + 1e-5)

    # ---------------------------------------------------------------- snapshot
    def snapshot(self) -> "RunningMeanStd":
        """A frozen copy of the current statistics.

        Used at the start of each SHAC learning episode so normalization is
        stable within the episode (see module docstring).
        """
        frozen = RunningMeanStd(self.mean.shape, device=self.device)
        frozen.mean = self.mean.clone()
        frozen.var = self.var.clone()
        frozen.count = self.count
        return frozen

    # ------------------------------------------------------------------- state
    def state_dict(self) -> dict:
        return {"mean": self.mean, "var": self.var, "count": self.count}

    def load_state_dict(self, d: dict) -> None:
        self.mean = d["mean"].clone()
        self.var = d["var"].clone()
        self.count = d["count"]
