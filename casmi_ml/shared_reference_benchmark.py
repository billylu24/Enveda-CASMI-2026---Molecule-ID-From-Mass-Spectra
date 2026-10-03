"""Actual supplied train: exact neutral/union rows and every sparse entry, fresh arms."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from baseline import ADDUCT_MASS, load_candidates, make_matrix
from casmi_ml.data import write_json
from casmi_ml.mass_candidates import mass_centers
from casmi_ml.metfrag import digest
from casmi_ml.reference_reuse import build_shared_reference
from casmi_ml.research_protocol import freeze


def run(output):
    output = Path(output)
    if output.exists() and any(
        p.name
        not in {
            "execution.log",
            "launch.lock",
            "job.lock",
            "training_budget.json",
            "training_budget.lock",
        }
        for p in output.iterdir()
    ):
        raise ValueError("Fresh benchmark directory required")
    output.mkdir(parents=True, exist_ok=True)
    train, test = Path("data/train.parquet"), Path("data/test.parquet")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "reference_reuse_sha256": digest("casmi_ml/reference_reuse.py"),
            "baseline_sha256": digest("baseline.py"),
            "train_sha256": digest(train),
            "test_sha256": digest(test),
            "scope": "Unlabeled actual400 reference runtime equivalence",
            "independent_acceptance": False,
            "new_training": False,
            "query_labels_used": False,
        },
    )
    frame = pd.read_parquet(test)
    neutral = frame.precursor_mz.to_numpy() - frame.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral)]
    centers = set(neutral.tolist())
    for _, query in frame.groupby("molecule_id", sort=False):
        centers.update(mass_centers(query, "charge_aware_union"))
    started = time.monotonic()
    legacy, _ = load_candidates(train, neutral)
    legacy_rows = pd.DataFrame(
        [r[:3] for r in legacy], columns=["mass", "inchikey14", "normalized_smiles"]
    )
    legacy_rows.to_parquet(output / "legacy.parquet", index=False)
    sparse.save_npz(output / "legacy.npz", make_matrix([r[3] for r in legacy]))
    del legacy
    union, _ = load_candidates(train, np.array(sorted(centers)))
    pd.DataFrame([r[:3] for r in union], columns=legacy_rows.columns).to_parquet(
        output / "union.parquet", index=False
    )
    sparse.save_npz(output / "union.npz", make_matrix([r[3] for r in union]))
    observed = {r[1] for r in union}
    del union
    baseline_seconds = time.monotonic() - started
    started = time.monotonic()
    rows, matrix, actual_keys = build_shared_reference(
        train, neutral, centers, output / "shared"
    )
    shared_seconds = time.monotonic() - started
    pd.testing.assert_frame_equal(
        legacy_rows, pd.DataFrame(rows, columns=legacy_rows.columns)
    )
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "union.parquet"),
        pd.read_parquet(output / "shared/rows.parquet"),
    )
    if actual_keys != observed:
        raise ValueError("Union observed members differ")
    counts = {}
    for name, actual in [
        ("legacy", matrix),
        ("union", sparse.load_npz(output / "shared/spectra.npz")),
    ]:
        expected = sparse.load_npz(output / (name + ".npz"))
        if actual.shape != expected.shape:
            raise ValueError("Sparse shape differs")
        for field in ["data", "indices", "indptr"]:
            if not np.array_equal(getattr(actual, field), getattr(expected, field)):
                raise ValueError("Sparse values/indices/order differ")
        counts[name] = {
            "rows": actual.shape[0],
            "nonzero": actual.nnz,
            "every_sparse_entry_exact": True,
        }
    report = {
        "valid": True,
        "diagnostic_only": True,
        "reference_counts": counts,
        "observed_members": len(observed),
        "seconds": {
            "baseline_two_scans": baseline_seconds,
            "shared_one_scan": shared_seconds,
        },
        "speedup": baseline_seconds / shared_seconds,
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "scope": "Reference construction equivalence only; cold full ranking and platform still required",
        "independent_acceptance": False,
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(p.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
