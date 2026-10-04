"""Local paired test of five individually supported high-route insertion slots."""

import argparse
import json
from functools import partial
from pathlib import Path
from unittest.mock import patch

from casmi_ml import full_gap_candidate_system as base
from casmi_ml.chembl_high_reference_prefix import reference_supported_prefix
from casmi_ml.data import write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze
from casmi_ml.strong_slot_inference import select_strong_slots

VARIANT = "high_strong_five_slots"


def select_slots(
    original,
    current163,
    proposed,
    critic,
    fragments,
    original_fragments,
    original_fallback,
    observed,
    *,
    slots,
):
    if slots not in (3, 5):
        raise ValueError("Frozen three-slot control or five-slot arm required")
    if slots == 3:
        return select_strong_slots(
            original,
            current163,
            proposed,
            critic,
            fragments,
            original_fragments,
            original_fallback,
            observed,
        )
    if original_fallback or any(k not in original_fragments for k in original[:3]):
        return current163, [], "strong_original_evidence_fallback"
    maximum = max(original_fragments[k] for k in original[:3])
    if fragments.get(proposed[0], 0) <= maximum:
        return current163, [], "strong_original_gate_fallback"
    supported = [
        k
        for k in proposed[:5]
        if fragments.get(k, 0) > maximum and critic[k] > critic[original[0]] + 0.05
    ]
    result = insert_generated(
        original, supported, reference_supported_prefix(original, observed), 5
    )
    return result, supported, "high_strong_slot_inserted"


def run(output, model, slots=3):
    output = Path(output)
    freeze(
        output / "slot_depth_protocol.json",
        {
            "source_sha256": digest(__file__),
            "base_source_sha256": digest(base.__file__),
            "original_selector_sha256": digest("casmi_ml/strong_slot_inference.py"),
            "audit_sha256": digest(
                "results/research_loop/candidate_stage_retention.json"
            ),
            "slots": slots,
            "rule": "Only existing high_fragment_inserted branch; original leader, informativeness, critic+.05, reference-protected original prefix and original first3 fragment maximum gates unchanged. For five-slot arm inspect first5 fragment-fused proposals and insert at most5 individually satisfying the same original fragment maximum and critic+.05. All low/fallback branches unchanged. No new physical candidates, scores, model fitting or threshold tuning.",
            "local_only": True,
            "independent_acceptance": False,
            "scope": "Complete cached repeated-development comparison; three-slot exact control must precede five-slot arm. No release activation or remote publication.",
        },
    )
    with patch(
        "casmi_ml.strong_slot_inference.select_strong_slots",
        new=partial(select_slots, slots=slots),
    ):
        report = base.run(output, model, control=slots == 3)
    if report is None:
        return None
    for mode in ("unknown", "known"):
        report[mode][VARIANT] = report[mode].pop("full_gap_candidate_compatibility")
        (output / f"{mode}_full_gap_candidate_compatibility.csv").rename(
            output / f"{mode}_{VARIANT}.csv"
        )
    report.update(
        slot_depth=slots,
        slot_depth_source_sha256=digest(__file__),
        diagnostic_only=True,
        deployment_required=slots != 3,
    )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--slots", type=int, choices=[3, 5], default=3)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
