"""Aggregate original label-free handoff versus final candidate retention diagnosis."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.cached_structure_keys import structure_key
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller

BASE = Path("artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay")
HANDOFF = Path("artifacts/research_loop/rounds/0206_full181_audited_exact_control")


def run(output):
    if not Controller().local_only():
        raise ValueError("Candidate handoff diagnosis is local-only")
    rankings = json.loads((BASE / "selected_rankings.json").read_text())
    cohorts = {}
    sources = {}
    for mode, expected in (("unknown", 2000), ("known", 1340)):
        legacy_file = BASE / f"{mode}_supported_leader_replay.csv"
        handoff_file = HANDOFF / mode / "baseline_full.json"
        rows = pd.read_csv(legacy_file).set_index("key")
        raw = json.loads(handoff_file.read_text())
        original = {
            r["molecule_id"]: [structure_key(s) for s in r["smiles"]] for r in raw
        }
        if (
            len(rows) != expected
            or rows.index.duplicated().any()
            or len(original) != len(raw)
            or not set(original).issubset(rows.index)
            or set(rankings[mode]) != set(rows.index)
            or not rows.covered.isin([0, 1]).all()
        ):
            raise ValueError("Complete aligned frozen cohorts required")
        available = set(rows.index[rows.covered > 0])
        before = {k for k, values in original.items() if k in values}
        after = {k for k, values in rankings[mode].items() if k in values}
        cohorts[mode] = {
            "molecules": expected,
            "legacy_candidate_availability_count": len(available),
            "original_handoff_rows": len(original),
            "original_handoff_missing_rows": expected - len(original),
            "truth_in_original_full_handoff": len(before),
            "truth_in_final_full_list": len(after),
            "truth_newly_added_by_external_path": len(after - before),
            "original_handoff_truth_removed_by_external_path": len(before - after),
            "legacy_available_but_not_in_original_handoff": len(available - before),
            "legacy_available_but_not_in_final_list": len(available - after),
            "original_handoff_truth_outside_legacy_available": len(before - available),
        }
        sources[mode] = {
            "legacy_metrics_sha256": digest(legacy_file),
            "original_handoff_sha256": digest(handoff_file),
        }
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "source_sha256": digest(__file__),
        "identity_source_sha256": digest("casmi_ml/cached_structure_keys.py"),
        "final_rankings_sha256": digest(BASE / "selected_rankings.json"),
        "input_sha256": sources,
        "cohorts": cohorts,
        "scope": "Completed repeated-development attribution only. Legacy covered is inherited availability; this traces original stored label-free generated/retrieval handoff and final complete236 lists. Does not determine a specific earlier gate or permit truth-based candidate selection. No new model, threshold, holdout or deployment.",
    }
    write_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
