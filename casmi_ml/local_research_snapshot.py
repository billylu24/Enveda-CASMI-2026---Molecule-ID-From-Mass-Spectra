"""Refresh safe local aggregate progress from authoritative controller state."""

import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.research_loop import Controller, now

ACTIVE = {"registered", "prepared", "running", "external_running", "queued"}


def refresh():
    c = Controller()
    state = c.read()
    if not c.local_only():
        raise ValueError("Local-only snapshot requires persistent local mode")
    active = [
        {
            "id": r["id"],
            "direction": r["direction"],
            "status": r["status"],
            "report": r["report"],
        }
        for r in state["rounds"]
        if r["status"] in ACTIVE
    ]
    active = [r for r in active if not Path(r["report"]).exists()]
    value = {
        "updated_at": now(),
        "local_only": True,
        "remote_publication": False,
        "best": state.get("development_best"),
        "local_verified_inference": state.get("local_verified_inference"),
        "active_rounds": active,
        "local_candidate_runtime": state.get("local_candidate_runtime"),
        "local_array_runtime": state.get("local_array_runtime"),
        "local_object_candidate_runtime": state.get("local_object_candidate_runtime"),
        "local_proposal_runtime": state.get("local_proposal_runtime"),
        "local_query_zero_control": state.get("local_query_zero_control"),
        "drivers": [
            "diagnostic_query_multispectrum_followup.log",
            "diagnostic_query_chemical_only_followup.log",
            "diagnostic_query_balanced_followup.log",
            "object_candidate_followup.log",
            "diagnostic_query_strong_peak_followup.log",
        ],
        "pending_sequence": "Complete274/275 balancing attribution and activate exact object runtime276/277. New optional strong-peak input needs280 source-bound zero control, then281/282 same-pair training and283/284 complete matched systems. Only complete frozen gate winners replace best.",
        "freeze_policy": "unknown absolute+.0005 and relative+2%, known drop<=.001/top1drop<=.005; current best, completed original0206 and historical181 protection; no new physical failed requests or budget fallback promotion",
        "runtime_claims": "Exact packed candidate and array-order all3340 cached rankings, proposal three full2000 dictionaries; no cold inference proof or independent acceptance",
        "independent_acceptance": False,
        "recent_rounds": [
            {
                "id": r["id"],
                "direction": r["direction"],
                "status": r["status"],
                "decision": r.get("decision"),
                "local_recorded": r.get("local_recorded", False),
            }
            for r in state["rounds"]
            if r["id"] >= "0244"
        ],
    }
    write_json("results/research_loop/local_continuation_manifest.json", value)
    path = Path("results/research_loop/local_diagnostic_cycle_progress.json")
    progress = json.loads(path.read_text()) if path.exists() else {}
    progress.update(
        updated_at=value["updated_at"],
        local_only=True,
        remote_publication=False,
        best_round=state["development_best"]["round"],
        active_rounds=[r["id"] for r in active],
        active_round=active[-1]["id"] if active else None,
        next_sequence=value["pending_sequence"],
        current_query_zero_control=state.get("local_query_zero_control"),
        local_verified_evidence_round=state["local_verified_inference"][
            "evidence_round"
        ],
    )
    progress["array_order_microbenchmark"]["scope"] = (
        "Sorting-segment microbenchmark followed by exact3340 cached inference control; activated, no accuracy or cold acceptance gain"
    )
    for name, filename in [
        ("multispectrum_attribution", "query_multispectrum_attribution.json"),
        ("chemical_only_attribution", "query_chemical_only_attribution.json"),
        ("balanced_attribution", "query_balanced_attribution.json"),
        ("strong_peak_attribution", "query_strong_peak_attribution.json"),
    ]:
        report = Path("results/research_loop") / filename
        if report.exists():
            progress[name] = json.loads(report.read_text())
    progress["proposal_runtime_control"] = {
        "complete": True,
        "full_query_matches_per_variant": 2000,
        "variants": ["native", "corrected_only", "corrected_fusion"],
        "accuracy_gain_claim": False,
        "enabled_for_explicit_local_proposal_entry": bool(
            state.get("local_proposal_runtime", {}).get("enabled")
        ),
    }
    write_json(path, progress)
    return value


def main():
    value = refresh()
    print(
        json.dumps(
            {"best": value["best"]["round"], "active": value["active_rounds"]}, indent=2
        )
    )


if __name__ == "__main__":
    main()
