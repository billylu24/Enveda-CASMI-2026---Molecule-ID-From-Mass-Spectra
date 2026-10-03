"""Fixed actual mass-window value/fingerprint comparison for object frame storage."""

import argparse
import gc
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.data import write_json
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.metfrag import digest
from casmi_ml.object_candidate_runtime import ObjectPackedCandidateIndex
from casmi_ml.packed_candidate_runtime import resolve
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Object-frame runtime benchmark is local-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    runtime = c.read()["local_candidate_runtime"]
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "trial_source_sha256": digest("casmi_ml/object_candidate_runtime.py"),
            "candidate_runtime": runtime,
            "sampling": "Same runtime-profile opaque SHA order100 nonempty baseline per scenario, all actual charge-aware union mass centers. No query outcomes used.",
            "intervention": "After unchanged packed-cache source validation, convert every string column from Arrow-backed storage to Python objects, same row values/order and float32 fingerprints. Isolated trial only.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    groups = {}
    for mode in ["unknown", "known"]:
        frame = pd.read_parquet(CONTROL / mode / "test.parquet")
        full = json.loads((CONTROL / mode / "baseline_full.json").read_text())
        ids = sorted(
            [r["molecule_id"] for r in full],
            key=lambda k: hashlib.sha256(("runtime-profile:" + k).encode()).digest(),
        )[:100]
        indexed = frame.groupby("molecule_id", sort=False).indices
        groups[mode] = [
            frame.iloc[indexed[k]].drop(columns=["molecule_id"]) for k in ids
        ]
    expected = {}
    timing = {mode: {} for mode in groups}
    for name in ["packed_arrow", "packed_object"]:
        constructor = (
            resolve(c, digest(DERIVED))
            if name == "packed_arrow"
            else lambda cat: ObjectPackedCandidateIndex(
                cat, runtime["store"], runtime["catalog_sha256"]
            )
        )
        started = time.monotonic()
        index = constructor(pd.read_parquet(DERIVED))
        initialization = time.monotonic() - started
        for mode, queries in groups.items():
            elapsed = 0.0
            count = 0
            for i, query in enumerate(queries):
                started = time.monotonic()
                pool, fps = candidate_window(index, query, "charge_aware_union")
                elapsed += time.monotonic() - started
                count += len(pool)
                identity = (mode, i)
                if name == "packed_arrow":
                    expected[identity] = (
                        pool.copy(),
                        np.packbits(fps.astype(np.uint8), axis=1),
                    )
                else:
                    baseline, packed = expected[identity]
                    pd.testing.assert_frame_equal(pool, baseline, check_dtype=False)
                    if not np.array_equal(
                        fps, np.unpackbits(packed, axis=1).astype(np.float32)
                    ):
                        raise ValueError("Object candidate float32 fingerprints differ")
            timing[mode][name] = {
                "queries": len(queries),
                "candidate_rows": count,
                "window_seconds": elapsed,
                "initialization_seconds": initialization,
            }
        del index
        gc.collect()
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "full_value_and_float32_fingerprint_windows_exact": 200,
        "timing": timing,
        "activated": False,
        "scope": "Candidate-window storage microbenchmark only; complete3340 inference required before activation",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
