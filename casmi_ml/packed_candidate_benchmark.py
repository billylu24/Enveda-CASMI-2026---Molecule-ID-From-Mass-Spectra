"""Matched observable candidate-window cache benchmark before full-rank activation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_cache import PackedCandidateIndex
from casmi_ml.ranking import CandidateIndex
from casmi_ml.research_protocol import freeze

STORE = Path("artifacts/research_loop/rounds/0250_local_packed_candidate_preparation")
VERIFIED = Path("artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay")


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((VERIFIED / "external_config.json").read_text())
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "cache_source_sha256": digest("casmi_ml/packed_candidate_cache.py"),
            "store_report_sha256": digest(STORE / "report.json"),
            "source_catalog_sha256": config["catalog_sha256"],
            "sampling": "Same fixed opaque100-per-mode runtime profile inputs; same supplied row order, windows and invalid structure policy.",
            "rule": "Check candidate DataFrame exact and all2048 float32 fingerprint values for every query, then report timings; no model inference, no labels or candidate scoring change.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    catalog = pd.read_parquet(config["catalog"])
    legacy = CandidateIndex(catalog)
    packed = PackedCandidateIndex(catalog, STORE, config["catalog_sha256"])
    counts = {}
    expected = {}
    source = Path("artifacts/research_loop/rounds/0249_local_ranking_runtime_profile")
    for mode in ["unknown", "known"]:
        test = pd.read_parquet(source / mode / "test.parquet")
        groups = test.groupby("molecule_id", sort=False)
        started = time.monotonic()
        for key, group in groups:
            if len(legacy.cache) > 20000:
                legacy.cache.clear()
            pool, fps = candidate_window(
                legacy, group.drop(columns=["molecule_id"]), "charge_aware_union"
            )
            expected[key] = (pool, fps)
        old_seconds = time.monotonic() - started
        started = time.monotonic()
        total = 0
        for key, group in groups:
            pool, fps = candidate_window(
                packed, group.drop(columns=["molecule_id"]), "charge_aware_union"
            )
            old_pool, old_fps = expected[key]
            pd.testing.assert_frame_equal(pool, old_pool)
            np.testing.assert_array_equal(fps, old_fps)
            total += len(pool)
        new_seconds = time.monotonic() - started
        counts[mode] = {
            "queries": len(groups),
            "candidate_fingerprint_rows": total,
            "every_candidate_and_float32_bit_exact": True,
            "legacy_seconds": old_seconds,
            "packed_seconds_including_exact_checks": new_seconds,
        }
        print(mode, counts[mode], flush=True)
        expected.clear()
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "verified": True,
        "counts": counts,
        "scope": "Matched cached windows200 opaque queries; includes equality checks in packed timing. Complete3340 full inference replay still required before local activation.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
