"""Paired exact high-route score ordering on the fixed opaque-query runtime sample."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.candidate_array_order import rank_values
from casmi_ml.chembl_fingerprint_prior import corrected_scores
from casmi_ml.data import write_json
from casmi_ml.inference import group_probability
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_cache import PackedCandidateIndex
from casmi_ml.research_protocol import freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def run(output):
    configure(42, threads=4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source = Path("artifacts/research_loop/rounds/0249_local_ranking_runtime_profile")
    config = json.loads(
        Path(
            "artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay/external_config.json"
        ).read_text()
    )
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "array_source_sha256": digest("casmi_ml/candidate_array_order.py"),
            "catalog_sha256": config["catalog_sha256"],
            "encoder_sha256": config["encoder_sha256"],
            "sampling": "Same fixed opaque100-per-mode profile sample, all nonempty mass windows; no label-selected subsets.",
            "rule": "Compute same original encoder group probability and corrected prior score vector once; compare old pandas scalar score/key sort and array lexsort completely, record ordering only runtime. No chemical rule, threshold, model or score arithmetic change.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    packed = PackedCandidateIndex(
        pd.read_parquet(config["catalog"]),
        "artifacts/research_loop/rounds/0250_local_packed_candidate_preparation",
        config["catalog_sha256"],
    )
    encoder, saved = load_deployment_checkpoint(config["encoder"], "scale")
    encoder.eval()
    prior = np.load(config["prior"])
    reports = {}
    for mode in ["unknown", "known"]:
        old_seconds = new_seconds = 0.0
        queries = candidates = 0
        test = pd.read_parquet(source / mode / "test.parquet")
        for _, group in test.groupby("molecule_id", sort=False):
            query = group.drop(columns=["molecule_id"])
            pool, fps = candidate_window(packed, query, "charge_aware_union")
            if not len(pool):
                continue
            probability = group_probability(encoder, query, saved["preprocessing"])
            values = corrected_scores(probability, fps, prior)
            started = time.monotonic()
            old = [
                pool.iloc[i].inchikey14
                for i in sorted(
                    range(len(pool)),
                    key=lambda i: (-values[i], pool.iloc[i].inchikey14),
                )
            ]
            old_seconds += time.monotonic() - started
            started = time.monotonic()
            new = rank_values(pool.inchikey14.tolist(), values)
            new_seconds += time.monotonic() - started
            if old != new:
                raise ValueError("Actual corrected full candidate ordering differs")
            queries += 1
            candidates += len(pool)
        reports[mode] = {
            "queries": queries,
            "candidate_rows": candidates,
            "complete_orders_exact": True,
            "scalar_sort_seconds": old_seconds,
            "array_sort_seconds": new_seconds,
        }
        print(mode, reports[mode], flush=True)
    reports.update(
        diagnostic_only=True,
        local_only=True,
        independent_acceptance=False,
        verified=True,
        scope="Fixed200 opaque-query actual corrected ranking order benchmark; complete full inference control required before activation",
    )
    write_json(output / "report.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
