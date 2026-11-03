"""
Minimal, dependency-free experiment logging.

Two outputs:
  1. <logdir>/metadata.json   -- the full configuration of the run
  2. <logdir>/log.csv         -- one row per learning episode

The CSV is intentionally simple so it can be inspected with pandas / Excel /
grep, and so `scripts/plot_training.py` can turn it into figures without any
additional logging infrastructure. If you prefer TensorBoard later, the
official implementation (NVlabs/DiffRL) logs the same quantities there.
"""

import csv
import json
import os
import time
from typing import Dict, Optional


class RunLogger:
    """Appends one CSV row per call to `log()`, flushing immediately so that
    training progress is visible on disk while the run is still going (e.g.
    `tail -f runs/<...>/log.csv` over SSH, or start plotting mid-run)."""

    def __init__(self, logdir: str, meta: Optional[Dict] = None):
        self.logdir = logdir
        os.makedirs(logdir, exist_ok=True)

        # Store the run configuration (e.g. the resolved dataclass configs
        # serialized via asdict) next to the CSV for reproducibility.
        if meta is not None:
            with open(os.path.join(logdir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)

        self._csv_path = os.path.join(logdir, "log.csv")
        self._file = None
        self._writer = None
        self._header_written = False
        self._t_start = time.time()

    # ------------------------------------------------------------------
    def log(self, row: Dict[str, object]) -> None:
        """Write one row. Missing keys (e.g. eval metrics between evals)
        become empty cells; new keys add columns lazily (padded retroactively
        is NOT supported -- so always pass the same key set)."""
        if self._file is None:
            self._file = open(self._csv_path, "w", newline="")
            self._writer = csv.DictWriter(self._file, fieldnames=list(row.keys()))
            self._writer.writeheader()
            self._header_written = True
        self._writer.writerow(row)
        self._file.flush()  # make it visible immediately

    # ------------------------------------------------------------------
    def elapsed(self) -> float:
        """Seconds since the logger was created (wall-clock training time)."""
        return time.time() - self._t_start

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
