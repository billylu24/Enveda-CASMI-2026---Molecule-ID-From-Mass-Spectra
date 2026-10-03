"""Reuse complete physical evaluation with explicitly bound count-score sources."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import casmi_ml.full_gap_candidate_system as base
from casmi_ml.candidate_count_compatibility import load_model
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

VARIANT = "candidate_count_compatibility"


def run(output, model, control=False):
    def count_freeze(path, value):
        value.update(
            source_sha256=digest(__file__),
            base_system_source_sha256=digest(base.__file__),
            original_compatibility_source_sha256=value["compatibility_source_sha256"],
            compatibility_source_sha256=digest(
                "casmi_ml/candidate_count_compatibility.py"
            ),
            count_source_sha256=digest("casmi_ml/candidate_chemistry_counts_audit.py"),
            multiplicity_source_sha256=digest(
                "casmi_ml/candidate_count_capacity_audit.py"
            ),
            variant=VARIANT,
            rule="Unchanged complete physical full-gap system/candidates/query/ref routes; original60 presence plus60 multiplicity extras observed_strength*max(n-1,0)/(n+1), weights sum<=2, original frozen critic and margin.05 retained. No dev-selected weights or holdout; all zero/mask/physical/frozen gates unchanged.",
        )
        freeze(path, value)

    with (
        patch.object(base, "load_model", new=load_model),
        patch.object(base, "freeze", new=count_freeze),
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
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--control", action="store_true")
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
