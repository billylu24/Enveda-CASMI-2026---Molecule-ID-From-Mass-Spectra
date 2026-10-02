"""Crash-conservative stage reservations and a process-wide single-GPU lock."""

import fcntl
import json
import math
import os
import time
from pathlib import Path

from casmi_ml.data import write_json


class StageBudget:
    def __init__(self, root, stage, run, requested, limit=86400):
        if not all(math.isfinite(v) and v > 0 for v in [requested, limit]):
            raise ValueError("Positive finite stage budget required")
        self.path = Path(root) / "training_budget.json"
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.stage, self.run, self.limit = stage, str(run), limit
        gpu_path = Path(
            os.environ.get("CASMI_GPU_LOCK", "artifacts/research_loop/gpu.lock")
        )
        gpu_path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = gpu_path.open("a")
        try:
            fcntl.flock(self.lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.started = time.monotonic()
            self.allowance = self._reserve(requested)
        except Exception:
            self.lock.close()
            raise

    def _transaction(self, operation):
        with self.path.with_suffix(".lock").open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            ledger = json.loads(self.path.read_text()) if self.path.exists() else {}
            entry = ledger.setdefault(
                self.stage,
                {"limit_seconds": self.limit, "used_seconds": 0.0, "runs": {}},
            )
            if entry["limit_seconds"] != self.limit:
                raise ValueError("Cannot change frozen stage budget")
            value = operation(entry)
            write_json(self.path, ledger)
            return value

    def _reserve(self, requested):
        def reserve(entry):
            allowance = max(0.0, min(requested, self.limit - entry["used_seconds"]))
            if allowance <= 0:
                raise RuntimeError(f"{self.stage} exhausted its stage budget")
            # Full reservation survives SIGKILL/power loss; only clean close refunds.
            entry["used_seconds"] += allowance
            entry["runs"][self.run] = entry["runs"].get(self.run, 0.0) + allowance
            entry["reservation"] = {
                "run": self.run,
                "seconds": allowance,
                "pid": os.getpid(),
                "started_at": time.time(),
            }
            return allowance

        return self._transaction(reserve)

    def checkpoint(self):
        return time.monotonic() - self.started < self.allowance

    def close(self):
        if not self.lock.closed:
            try:
                elapsed = min(self.allowance, max(0.0, time.monotonic() - self.started))
                refund = self.allowance - elapsed

                def finish(entry):
                    entry["used_seconds"] -= refund
                    entry["runs"][self.run] -= refund
                    entry.pop("reservation", None)

                self._transaction(finish)
            finally:
                fcntl.flock(self.lock, fcntl.LOCK_UN)
                self.lock.close()

    def __del__(self):
        if hasattr(self, "lock") and hasattr(self, "allowance"):
            self.close()
