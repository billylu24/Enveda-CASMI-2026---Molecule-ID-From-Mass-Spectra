"""Benchmark and fully verify exact candidate-cache runtime before local activation."""

import fcntl
import json

from casmi_ml.data import write_json
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.diagnostic_system_followup import wait_for_report
from casmi_ml.packed_candidate_benchmark import STORE
from casmi_ml.packed_candidate_runtime import activate
from casmi_ml.research_loop import Controller


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Packed candidate followup is local-only")
    with (c.root / "packed_candidate_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not wait_for_report(c, STORE.name):
            return
        c.evaluate_round(STORE.name)
        c.complete_round(STORE.name)
        execute(
            c,
            "0251_packed_candidate_window_benchmark",
            "casmi_ml.packed_candidate_benchmark",
            [],
            "runtime_diagnostic",
        )
        source, _ = execute(
            c,
            "0252_packed_candidate_full_inference_control",
            "casmi_ml.packed_candidate_verification",
            [],
            "implementation_control",
        )
        report = json.loads((source / "report.json").read_text())
        if not report["verified"]:
            raise RuntimeError(
                "Packed runtime control incomplete; retain original runtime"
            )
        activate(c, source, STORE)
        write_json(
            "results/research_loop/packed_candidate_cycle.json",
            {
                "local_only": True,
                "status": "complete",
                "independent_acceptance": False,
                "remote_publication": False,
                "exact_full_rankings": 3340,
                "scope": "Candidate preparation runtime only; no accuracy or cold inference claim",
            },
        )


if __name__ == "__main__":
    main()
