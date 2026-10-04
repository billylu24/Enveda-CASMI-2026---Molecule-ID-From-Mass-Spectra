"""Prove cached train subsets match original rows, order and every sparse entry."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

from baseline import ADDUCT_MASS
from casmi_ml.data import write_json
from casmi_ml.mass_candidates import mass_centers
from casmi_ml.metfrag import digest
from casmi_ml.reference_reuse import build_shared_reference
from casmi_ml.research_protocol import freeze
from casmi_ml.train_reference_cache import build_cached_reference


def run(output, store):
    output, store = Path(output), Path(store)
    train, test = Path("data/train.parquet"), Path("data/test.parquet")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "cache_source_sha256": digest("casmi_ml/train_reference_cache.py"),
            "train_sha256": digest(train),
            "test_sha256": digest(test),
            "store_report_sha256": digest(store / "report.json"),
            "query_labels_used": False,
            "rule": "Full actual400 reference subset equality against original one-scan construction: all legacy/union rows, sparse data/indices/indptr and observed keys. Cache preparation uses entire train only; targets enter only unchanged runtime subset filter. Runtime-equivalence test, no scientific acceptance.",
        },
    )
    frame = pd.read_parquet(test)
    neutral = frame.precursor_mz.to_numpy() - frame.adduct.map(ADDUCT_MASS).to_numpy()
    neutral = neutral[np.isfinite(neutral)]
    centers = set(neutral.tolist())
    for _, query in frame.groupby("molecule_id", sort=False):
        centers.update(mass_centers(query, "charge_aware_union"))
    started = time.monotonic()
    expected, matrix, observed = build_shared_reference(
        train, neutral, centers, output / "original"
    )
    original_seconds = time.monotonic() - started
    started = time.monotonic()
    actual, cached, keys = build_cached_reference(
        train, neutral, centers, output / "cached", store=store
    )
    cached_seconds = time.monotonic() - started
    if expected != actual or observed != keys:
        raise ValueError("Legacy rows/order or complete observed membership changed")
    pd.testing.assert_frame_equal(
        pd.read_parquet(output / "original/rows.parquet"),
        pd.read_parquet(output / "cached/rows.parquet"),
    )
    counts = {}
    for name, a, b in [
        ("legacy", matrix, cached),
        (
            "union",
            sparse.load_npz(output / "original/spectra.npz"),
            sparse.load_npz(output / "cached/spectra.npz"),
        ),
    ]:
        if a.shape != b.shape or any(
            not np.array_equal(getattr(a, k), getattr(b, k))
            for k in ("data", "indices", "indptr")
        ):
            raise ValueError("Original sparse reference values or ordering changed")
        counts[name] = {
            "rows": a.shape[0],
            "nonzero": a.nnz,
            "every_sparse_entry_exact": True,
        }
    result = {
        "valid": True,
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "reference_counts": counts,
        "observed_members": len(observed),
        "seconds": {
            "original_scan": original_seconds,
            "query_independent_train_cache": cached_seconds,
        },
        "source_sha256": digest(__file__),
        "cache_source_sha256": digest("casmi_ml/train_reference_cache.py"),
        "store_report_sha256": digest(store / "report.json"),
        "scope": "Full400 library construction exactness only; no query labels, packaged query subsets or prediction cache. Fresh75+cold400+platform400 remain required before any release.",
    }
    write_json(output / "report.json", result)
    print(json.dumps(result), flush=True)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--store", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
