"""Full training-library membership equivalence and materialization runtime."""

import argparse
import gc
import hashlib
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd

from baseline import ADDUCT_MASS, load_candidates
from casmi_ml.data import write_json
from casmi_ml.mass_candidates import mass_centers
from casmi_ml.metfrag import digest
from casmi_ml.observed_reference_keys import load_observed_keys
from casmi_ml.research_protocol import freeze


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    queries = pd.read_parquet("data/test.parquet")
    forbidden = {"inchikey14", "normalized_smiles", "molecular_formula", "fingerprint"}
    if forbidden & set(queries.columns):
        raise ValueError("Actual unlabeled input required")
    neutral = (
        queries.precursor_mz.to_numpy() - queries.adduct.map(ADDUCT_MASS).to_numpy()
    )
    centers = set(neutral[np.isfinite(neutral)].tolist())
    for _, group in queries.groupby("molecule_id", sort=False):
        centers.update(mass_centers(group, "charge_aware_union"))
    centers = np.array(sorted(centers))
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(Path(__file__)),
            "baseline_sha256": digest("baseline.py"),
            "membership_source_sha256": digest("casmi_ml/observed_reference_keys.py"),
            "train_sha256": digest("data/train.parquet"),
            "unlabeled_test_sha256": digest("data/test.parquet"),
            "rule": "Exact all observable adaptive-prefix mass centers. Original load_candidates then observed key set vs streaming nonempty criterion keys. Same original formula/ppm/peak-quality and35ppm masswindow. No ranking/physics changes. Full training library, no test labels. Timing scoped CPU materialization; new membership cannot deploy unless exact.",
            "target_mass_centers": len(centers),
            "independent_acceptance": False,
            "new_accuracy_claim": False,
        },
    )
    started = time.monotonic()
    records, _ = load_candidates("data/train.parquet", centers)
    expected = {r[1] for r in records}
    original_seconds = time.monotonic() - started
    spectra = len(records)
    del records
    gc.collect()
    started = time.monotonic()
    actual = load_observed_keys("data/train.parquet", centers)
    new_seconds = time.monotonic() - started
    if actual != expected:
        write_json(
            output / "private_key_difference.json",
            {
                "missing": sorted(expected - actual),
                "additional": sorted(actual - expected),
            },
        )
        raise ValueError("Full original library membership differs")
    report = {
        "diagnostic_only": True,
        "molecules": int(queries.molecule_id.nunique()),
        "reference_spectra": spectra,
        "reference_keys": len(actual),
        "full_key_set_exact": True,
        "target_mass_centers": len(centers),
        "key_set_sha256": hashlib.sha256(
            json.dumps(sorted(actual)).encode()
        ).hexdigest(),
        "seconds": {"original": original_seconds, "membership": new_seconds},
        "speedup": original_seconds / new_seconds,
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "independent_acceptance": False,
        "unlabeled_input": True,
        "scope": "Runtime-only full-library membership, no accuracy improvement; timings include materialization and page cache may differ between arms.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))


if __name__ == "__main__":
    main()
