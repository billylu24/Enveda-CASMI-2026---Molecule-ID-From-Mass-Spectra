"""Finish isolated exact object candidate controls and activate after science arms."""

import fcntl
import json

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.diagnostic_system_followup import wait_for_report
from casmi_ml.local_research_snapshot import refresh
from casmi_ml.local_verified_extension import verified_config
from casmi_ml.object_candidate_activation import activate
from casmi_ml.research_loop import Controller


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Object candidate followup is local-only")
    with (c.root / "object_candidate_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        for rid in [
            "0276_object_candidate_full_control",
            "0277_object_candidate_proposal_control",
        ]:
            if not wait_for_report(c, rid):
                return
            c.evaluate_round(rid)
            c.complete_round(rid)
        if wait_for_complete_system(c, "0275_query_balanced_mask_system") is None:
            return
        full = c.root / "rounds/0276_object_candidate_full_control"
        protocol = json.loads((full / "protocol.json").read_text())
        if protocol["incumbent_proof"] != c.read()["local_verified_inference"]:
            full, _ = execute(
                c,
                "0278_object_candidate_current_best_control",
                "casmi_ml.object_candidate_verification",
                [],
                "runtime_diagnostic",
            )
        activate(c, full, c.root / "rounds/0277_object_candidate_proposal_control")
        verified_config(c)
        refresh()
        print(
            "Object candidate storage exact local runtime active", full.name, flush=True
        )


if __name__ == "__main__":
    main()
