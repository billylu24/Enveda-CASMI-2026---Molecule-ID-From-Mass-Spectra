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
        "local_blas_runtime": state.get("local_blas_runtime"),
        "local_validation_runtime": state.get("local_validation_runtime"),
        "local_proposal_runtime": state.get("local_proposal_runtime"),
        "local_query_zero_control": state.get("local_query_zero_control"),
        "drivers": [
            "diagnostic_query_multispectrum_followup.log",
            "diagnostic_query_chemical_only_followup.log",
            "diagnostic_query_balanced_followup.log",
            "object_candidate_followup.log",
            "diagnostic_query_strong_peak_followup.log",
            "structure_key_followup_v2.log",
            "diagnostic_query_admission_followup_v2.log",
            "diagnostic_query_all_spectra_followup.log",
            "candidate_compatibility_followup.log",
            "extended_candidate_followup.log",
            "intensity_candidate_followup.log",
            "max_candidate_followup.log",
            "full_gap_candidate_followup_v2.log",
            "candidate_count_followup.log",
            "connected_candidate_followup.log",
            "aromatic_h_candidate_followup.log",
            "high_slot_depth_followup.log",
        ],
        "pending_sequence": "Scientific236/evidence286 retained.319 exact3340 validation cache active.337count+.436%,344connected-ion+.189%,351aromaticH+.227% all rejected with complete matching zero/mask/physical controls. Candidate handoff and3340exact stage audit:unknown248 original truths retained, external adds33/removes0; inherited covered397 equals original+all corrected proposals, not retrieval coverage.149 available truths absent at original handoff:133 outside chosen retrieval/generation,16 selected retrieval-pool-only.116 absent finally:44 outside actual100,72 within100.355exact3slotcontrol precedes356frozen5slotindividual-gate test, no source activation.353 prefetch control failed missing model path, retained;354 corrected all3340 exact same-science two-JVM control passed0budget/newfailure; fresh75 fullranks exact and57physical dictionaries equal serial. New prefetch cold400473.84s/5614MiB/zero budget and400Top25/265rawdicts exact vs serial; source-boundfresh75exact60.2s. Renewed private upload/platform pending. Separate requested one236 competition submission still unconsumed: original/portable platform105 physical budget fallbacks, packed pandas incompatibility fixed. New prefetch cold400/fresh75/platform400 proof required; persistent local_only and no GitHub push.",
        "freeze_policy": "unknown absolute+.0005 and relative+2%, known drop<=.001/top1drop<=.005; current best, completed original0206 and historical181 protection; no new physical failed requests or budget fallback promotion",
        "runtime_claims": "Exact packed/object/array all3340 cached ranks and proposal three full2000 dictionaries. Separate scientific236 one-time release cold400 resource/physical/ranking verified; no paired whole-pipeline speed or independent accuracy claim.",
        "one_time_kaggle": {
            key: state.get("one_time_kaggle", {}).get(key)
            for key in (
                "identity",
                "status",
                "remote",
                "submission_id",
                "public_score",
                "automatic_future_submissions",
            )
        },
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
        ("admission_margin_attribution", "query_admission_attribution.json"),
        ("all_spectra_attribution", "query_all_spectra_attribution.json"),
        (
            "candidate_compatibility_attribution",
            "candidate_compatibility_attribution.json",
        ),
        ("extended_candidate_attribution", "extended_candidate_attribution.json"),
        ("intensity_candidate_attribution", "intensity_candidate_attribution.json"),
        ("max_candidate_attribution", "max_candidate_attribution.json"),
        ("full_gap_candidate_attribution", "full_gap_candidate_attribution.json"),
        ("candidate_count_attribution", "candidate_count_attribution.json"),
        ("connected_candidate_attribution", "connected_candidate_attribution.json"),
        ("aromatic_h_candidate_attribution", "aromatic_h_candidate_attribution.json"),
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
