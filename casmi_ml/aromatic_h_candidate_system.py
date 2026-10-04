"""Complete physical evaluation with explicitly bound connected-ion score sources."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import casmi_ml.full_gap_candidate_system as base
from casmi_ml.aromatic_h_candidate_compatibility import load_model
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

VARIANT = "aromatic_h_candidate_compatibility"


def run(output, model, control=False):
    def connected_freeze(path, value):
        value.update(
            source_sha256=digest(__file__),
            base_system_source_sha256=digest(base.__file__),
            original_compatibility_source_sha256=value["compatibility_source_sha256"],
            compatibility_source_sha256=digest(
                "casmi_ml/aromatic_h_candidate_compatibility.py"
            ),
            hydrogen_source_sha256=digest("casmi_ml/fragment_hydrogen_audit.py"),
            refinement_source_sha256=digest("casmi_ml/aromatic_h_capacity_audit.py"),
            variant=VARIANT,
            rule="Same complete original physical system; only3 aromatic observed diagnostic-ion channels refined by single-bond connected heavy composition and exact original H. Loss channels/60D/critic/mean/sqrt/2.0bonus/L2.01/margin.05 unchanged. No newly opened holdout or automatic publication.",
        )
        freeze(path, value)

    with (
        patch.object(base, "load_model", new=load_model),
        patch.object(base, "freeze", new=connected_freeze),
    ):
        report = base.run(output, model, control)
    if report is None:
        return None
    output = Path(output)
    for mode in ("unknown", "known"):
        report[mode][VARIANT] = report[mode].pop("full_gap_candidate_compatibility")
        (output / f"{mode}_full_gap_candidate_compatibility.csv").rename(
            output / f"{mode}_{VARIANT}.csv"
        )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--control", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
