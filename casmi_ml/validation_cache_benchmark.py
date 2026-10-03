"""Paired validation-only benchmark over every actual input and final ranking CSV."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from casmi_ml.cached_submission_validation import (
    checked_structure_key,
    validate_submission,
)
from casmi_ml.data import write_json
from casmi_ml.inference import validate_submission as original
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

SOURCE = Path("artifacts/research_loop/rounds/0286_structure_identity_full_control")


def run(output):
    if not Controller().local_only():
        raise ValueError("Validation benchmark is local-only")
    output = Path(output)
    input_files = {}
    data = []
    for mode in ["unknown", "known"]:
        test = pd.read_parquet(CONTROL / mode / "test.parquet")
        for kind, path in [
            ("input", CONTROL / mode / "baseline.csv"),
            ("output", SOURCE / mode / "selected.csv"),
        ]:
            input_files[mode + ":" + kind] = digest(path)
            data.append((mode, kind, test, pd.read_csv(path)))
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "cache_source_sha256": digest("casmi_ml/cached_submission_validation.py"),
            "validator_source_sha256": digest("casmi_ml/inference.py"),
            "actual_input_sha256": input_files,
            "rule": "Every original nonempty baseline and actual verified best output CSV, unknown/known, same source exception messages and per-row identity/count/duplicate checks. Alternate legacy->cold cache then cold cache->legacy, preserve actual full mode/input/output order.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    timings = []
    for order in [("legacy", "cache"), ("cache", "legacy")]:
        for name in order:
            checked_structure_key.cache_clear()
            started = time.monotonic()
            count = 0
            for mode, kind, test, submission in data:
                if Controller().stopped():
                    return None
                fn = original if name == "legacy" else validate_submission
                fn(test, submission)
                count += len(submission)
            timings.append(
                {
                    "implementation": name,
                    "seconds": time.monotonic() - started,
                    "validated_rows": count,
                    "cache_info": checked_structure_key.cache_info()._asdict()
                    if name == "cache"
                    else None,
                }
            )
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "verified_all_actual_input_and_output_rows": all(
            x["validated_rows"] == 6670 for x in timings
        ),
        "timings": timings,
        "activated": False,
        "scope": "Pure per-SMILES validity and connectivity memoization; validation-only paired timing, full3340 inference exact control still required; no scientific or cold pipeline gain",
    }
    if not report["verified_all_actual_input_and_output_rows"]:
        raise ValueError(
            "Expected all1995+1340 actual nonempty rows twice for input/output"
        )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
