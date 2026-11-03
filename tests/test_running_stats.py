"""
Quick test of the running mean/variance used for observation normalization
(the parallel-variance merge of Chan et al., as in the official code).

Run directly:  uv run python tests/test_running_stats.py
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch

from shac.utils.running_stats import RunningMeanStd


def test_matches_batch_statistics():
    """Statistics after ingesting a stream must equal the statistics of the
    whole stream computed in one batch."""
    torch.manual_seed(0)
    rms = RunningMeanStd(shape=(5,), device="cpu")
    stream = []
    for _ in range(20):
        batch = torch.randn(37, 5)
        rms.update(batch)
        stream.append(batch)
    all_data = torch.cat(stream, dim=0)

    err_mean = (rms.mean - all_data.mean(dim=0)).abs().max().item()
    err_var = (rms.var - all_data.var(dim=0, unbiased=False)).abs().max().item()
    assert err_mean < 1e-4 and err_var < 1e-3, (err_mean, err_var)
    print(f"  running stats match batch stats "
          f"(mean err {err_mean:.2e}, var err {err_var:.2e})")


def test_snapshot_is_frozen():
    """The frozen snapshot must not change when the live stats update."""
    rms = RunningMeanStd(shape=(3,), device="cpu")
    rms.update(torch.randn(100, 3))
    snap = rms.snapshot()
    before = snap.mean.clone()
    rms.update(torch.randn(1000, 3) + 10.0)
    assert torch.equal(snap.mean, before), "snapshot changed unexpectedly"
    print("  snapshot stays frozen while live stats update (OK)")


if __name__ == "__main__":
    test_matches_batch_statistics()
    test_snapshot_is_frozen()
    print("ALL RUNNING-STATS TESTS PASSED")
