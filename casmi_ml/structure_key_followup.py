"""Defer current-best identity source amendment until paired science is finalized."""

import fcntl
import json
from pathlib import Path

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.local_research_snapshot import refresh
from casmi_ml.local_verified_extension import verified_config
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.structure_key_activation import activate


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Identity cache followup is local-only")
    with (c.root / "structure_key_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if wait_for_complete_system(c, "0284_query_strong_peak_mask_system") is None:
            return
        source, _ = execute(
            c,
            "0286_structure_identity_full_control",
            "casmi_ml.structure_key_full_replay",
            [],
            "runtime_diagnostic",
        )
        activate(c, source)
        verified_config(c)
        amendment = json.loads(
            Path(
                "artifacts/research_loop/structure_key_source_trial/system_amendment.json"
            ).read_text()
        )
        if (
            digest("casmi_ml/diagnostic_query_system.py")
            != amendment["original_source_sha256"]
            or digest(amendment["amended_source"]) != amendment["amended_source_sha256"]
        ):
            raise ValueError("Query identity runtime source amendment changed")
        Path("casmi_ml/diagnostic_query_system.py").write_bytes(
            Path(amendment["amended_source"]).read_bytes()
        )
        source, _ = execute(
            c,
            "0287_query_identity_runtime_zero_control",
            "casmi_ml.diagnostic_query_system",
            [
                "--critic",
                str(
                    c.root / "rounds/0280_query_strong_peak_zero_control/zero_model.pt"
                ),
                "--control",
            ],
            "implementation_control",
        )
        if not json.loads((source / "report.json").read_text())["control_passed"]:
            raise ValueError("Current source query zero control failed")
        c.change(lambda state: state.update(local_query_zero_control=str(source)))
        refresh()
        print(
            "Exact local structure identity runtime and current query zero verified",
            flush=True,
        )


if __name__ == "__main__":
    main()
