"""Complete original physical236 system with conditional diagnostic candidate support."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml import full_gap_candidate_system as base
from casmi_ml.data import write_json
from casmi_ml.energy_conditioned_compatibility import load_model
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

VARIANT = "energy_conditioned_compatibility"


def run(output, model, control=False):
    def bound_freeze(path, value):
        value.update(
            source_sha256=digest(__file__),
            base_system_source_sha256=digest(base.__file__),
            conditional_loader_source_sha256=digest(
                "casmi_ml/energy_conditioned_compatibility.py"
            ),
            variant=VARIANT,
            rule="Original236 full physical system;120D original60 binary plus60 per-query measured-row observed diagnostic*predicted candidate strength conditioned on same row singletonCE/adduct/instrument. Missing/mergedCE extras0, originalbinary untouched. Original total2/margin.05/windows/physical1200 unchanged. Complete zero/mask exact3340 and both full systems before scientific evaluation, local-only.",
        )
        freeze(path, value)

    with (
        patch.object(base, "load_model", new=load_model),
        patch.object(base, "freeze", new=bound_freeze),
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
