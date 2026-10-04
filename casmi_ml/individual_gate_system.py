"""Local high-route experiment: qualify candidates before selecting three slots."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml import full_gap_candidate_system as base
from casmi_ml.chembl_fragment_pilot import informative_fragments
from casmi_ml.chembl_high_reference_prefix import reference_supported_prefix
from casmi_ml.chembl_routed_inference import select_high_fragment as original_high
from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze
from casmi_ml.strong_slot_inference import select_strong_slots as original_strong

VARIANT = "individual_high_candidate_gates"


def select_high(current, tail, candidates, critic, fragments, fallback):
    if fallback:
        return original_high(current, tail, candidates, critic, fragments, fallback)
    proposed = rerank(
        candidates, {}, [], 0.5, top_n=len(candidates), fragment_scores=fragments
    )
    if not informative_fragments(proposed, fragments):
        return tail, candidates[:3], "high_fragment_evidence_keep_tail"
    supported = [
        k
        for k in proposed
        if fragments.get(k, 0) > fragments.get(current[0], 0)
        and critic[k] > critic[current[0]] + 0.05
    ]
    if not supported:
        return current, [], "high_fragment_remove_tail"
    return (
        insert_generated(current, supported, 3, 3),
        supported[:3],
        "high_fragment_inserted",
    )


def select_strong(
    original,
    current163,
    proposed,
    critic,
    fragments,
    original_fragments,
    original_fallback,
    observed,
):
    if original_fallback or any(k not in original_fragments for k in original[:3]):
        return current163, [], "strong_original_evidence_fallback"
    maximum = max(original_fragments[k] for k in original[:3])
    supported = [
        k
        for k in proposed
        if fragments.get(k, 0) > maximum and critic[k] > critic[original[0]] + 0.05
    ][:3]
    if not supported:
        return current163, [], "strong_original_gate_fallback"
    return (
        insert_generated(
            original, supported, reference_supported_prefix(original, observed), 3
        ),
        supported,
        "high_strong_slot_inserted",
    )


def run(output, model, control=False):
    output = Path(output)
    freeze(
        output / "individual_gate_protocol.json",
        {
            "source_sha256": digest(__file__),
            "base_source_sha256": digest(base.__file__),
            "trace_sha256": digest("results/research_loop/candidate_gate_trace.json"),
            "control": control,
            "rule": "Frozen high-route shortlist100, same whole-pool informativeness, critic margin.05, positive physical support over original first, original first3 physical maximum and reference prefix. Filter candidates individually before choosing at most3 slots; no veto from an unqualified first fused proposal. Strong selector scans qualified original100 rather than only first3, then retains at most3. Low, critic-query-possible and physical-fallback branches unchanged. New original-first3 physical requests may be required; no new candidate/threshold/weight. Must complete both cohorts with no new physical failure/budget fallback.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with (
        patch(
            "casmi_ml.chembl_routed_inference.select_high_fragment",
            new=original_high if control else select_high,
        ),
        patch(
            "casmi_ml.strong_slot_inference.select_strong_slots",
            new=original_strong if control else select_strong,
        ),
    ):
        report = base.run(output, model, control=control)
    if report is None:
        return None
    for mode in ("unknown", "known"):
        report[mode][VARIANT] = report[mode].pop("full_gap_candidate_compatibility")
        (output / f"{mode}_full_gap_candidate_compatibility.csv").rename(
            output / f"{mode}_{VARIANT}.csv"
        )
    report.update(
        diagnostic_only=True,
        individual_gate_source_sha256=digest(__file__),
        deployment_required=not control,
    )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--control", action="store_true")
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
