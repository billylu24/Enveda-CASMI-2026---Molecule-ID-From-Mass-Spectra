"""All observable development spectra exact audit of maximum-channel extraction."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.diagnostic_feature_runtime import maximum_evidence_features
from casmi_ml.diagnostic_residual import legacy_evidence_features
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, freeze


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    path = ROOT / "researchdev.parquet"
    frame = pd.read_parquet(
        path,
        columns=[
            "adduct",
            "ionization_mode",
            "instrument_type",
            "precursor_mz",
            "ms2_mzs",
            "ms2_normalized_intensities",
        ],
    )
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "legacy_source_sha256": digest("casmi_ml/diagnostic_residual.py"),
            "maximum_source_sha256": digest("casmi_ml/diagnostic_feature_runtime.py"),
            "observable_source_sha256": digest(path),
            "spectra": len(frame),
            "arms": ["coarse", "precise"],
            "rule": "Each observable row24-dimensional float32 features must match legacy peak-edge enumeration bit-for-bit under both tolerances. No labels, new training, model or ranking change.",
            "local_only": True,
            "diagnostic_only": True,
        },
    )
    stats = {}
    rows = frame.to_dict("records")
    for precise in (False, True):
        legacy_seconds = maximum_seconds = 0.0
        nonempty = 0
        for i, row in enumerate(rows):
            start = time.monotonic()
            expected = legacy_evidence_features(row, precise)
            legacy_seconds += time.monotonic() - start
            start = time.monotonic()
            actual = maximum_evidence_features(row, precise)
            maximum_seconds += time.monotonic() - start
            if not np.array_equal(expected, actual):
                raise ValueError(
                    "Maximum diagnostic channels differ from original observable feature extraction"
                )
            nonempty += bool(actual.any())
            if (i + 1) % 1000 == 0:
                print(
                    "feature exact",
                    "precise" if precise else "coarse",
                    i + 1,
                    flush=True,
                )
        stats["precise" if precise else "coarse"] = {
            "spectra": len(rows),
            "exact_features": len(rows),
            "nonempty": nonempty,
            "legacy_seconds": legacy_seconds,
            "maximum_seconds": maximum_seconds,
        }
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "verified": True,
        "arms": stats,
        "scope": "Observable feature extraction equivalence and shared-machine microbenchmark only; no accuracy or cold full-system claim",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
