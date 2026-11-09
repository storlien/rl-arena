"""
Minimal, dependency-free experiment logging.

Two outputs:
  1. <logdir>/metadata.json   -- the full configuration of the run
  2. <logdir>/log.csv         -- one row per learning episode

The CSV is intentionally simple so it can be inspected with pandas / Excel /
grep, and so `scripts/plot_training.py` can turn it into figures without any
additional logging infrastructure. If you prefer TensorBoard later, the
official implementation (NVlabs/DiffRL) logs the same quantities there.

Safety semantics:
  * log.csv is NEVER silently overwritten: logging into a logdir that
    already has a log.csv raises, so re-pointing a run at an existing
    directory cannot destroy its history.
  * append=True (used by --resume) continues an existing log.csv instead:
    rows are appended after the existing ones, no header is re-written.
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

    def __init__(self, logdir: str, meta: Optional[Dict] = None,
                 append: bool = False):
        self.logdir = logdir
        self._append = append
        os.makedirs(logdir, exist_ok=True)

        # Store the run configuration (e.g. the resolved dataclass configs
        # serialized via asdict) next to the CSV for reproducibility.
        if meta is not None:
            with open(os.path.join(logdir, "metadata.json"), "w") as f:
                json.dump(meta, f, indent=2, default=str)

        self._csv_path = os.path.join(logdir, "log.csv")
        self._file = None
        self._writer = None
        self._t_start = time.time()

    # ------------------------------------------------------------------
    def log(self, row: Dict[str, object]) -> None:
        """Write one row. The CSV schema (column set) is fixed by the FIRST
        row; every later row must pass the same key set -- None values are
        written as empty cells (e.g. eval metrics between evals). A row
        with drifted keys raises a clear error instead of csv's cryptic
        one; new diagnostics therefore require a fresh logdir."""
        if self._file is None:
            exists = os.path.exists(self._csv_path)
            if exists and not self._append:
                raise FileExistsError(
                    f"{self._csv_path} already exists -- refusing to "
                    "overwrite a run's history. Use a fresh --logdir (or "
                    "RunLogger(..., append=True) / --resume to continue an "
                    "existing run).")
            mode = "a" if (exists and self._append) else "w"
            self._file = open(self._csv_path, mode, newline="")
            self._writer = csv.DictWriter(self._file,
                                          fieldnames=list(row.keys()))
            if mode == "w":
                self._writer.writeheader()
        try:
            self._writer.writerow(row)
        except ValueError as e:
            raise ValueError(
                f"log row keys drifted from this CSV's header (fixed at the "
                f"first row): {e}. Use a fresh logdir for new diagnostics."
            ) from e
        self._file.flush()  # make it visible immediately

    # ------------------------------------------------------------------
    def elapsed(self) -> float:
        """Seconds since the logger was created (wall-clock training time)."""
        return time.time() - self._t_start

    def close(self) -> None:
        if self._file is not None:
            self._file.close()
            self._file = None
            self._writer = None

    def __enter__(self) -> "RunLogger":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
