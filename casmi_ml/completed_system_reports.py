"""Resolve the finalized complete-evidence round, including bounded cache retries."""

import json
import time
from pathlib import Path


def completed_system_report(c, rid):
    rounds = {row["id"]: row for row in c.read()["rounds"]}
    for attempt in range(3):
        trial = rid if attempt == 0 else rid + f"_cache_completion{attempt}"
        row = rounds.get(trial, {})
        if row.get("status") == "failed":
            raise RuntimeError("Required local system failed: " + trial)
        # A file may appear before evaluation and completion update controller
        # state. Do not start a dependent arm on that transient report.
        if not row.get("local_recorded"):
            return None
        source = Path(row["report"])
        report = json.loads(source.read_text())
        if (
            not report["development_budget_fallbacks"]
            and not report["incremental_new_failure_contexts"]
        ):
            return source.parent
    raise RuntimeError("All bounded system retries have incomplete evidence: " + rid)


def wait_for_complete_system(c, rid):
    while not c.stopped():
        source = completed_system_report(c, rid)
        if source is not None:
            return source
        time.sleep(10)
    return None
