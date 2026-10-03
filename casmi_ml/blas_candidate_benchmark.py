"""Compare BLAS thread overhead on fixed actual candidate-score windows."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_info, threadpool_limits

from casmi_ml.chembl_fingerprint_prior import corrected_scores
from casmi_ml.data import write_json
from casmi_ml.generated_native_scores import native_scores
from casmi_ml.inference import group_probability
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_runtime import resolve
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

CONTROL_SOURCE = Path(
    "artifacts/research_loop/rounds/0286_structure_identity_full_control"
)


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("BLAS score microbenchmark is local-only")
    output = Path(output)
    config = json.loads((CONTROL_SOURCE / "external_config.json").read_text())
    configure(threads=4)
    info = [x for x in threadpool_info() if x["user_api"] == "blas"]
    if not info or any(x["num_threads"] != 24 for x in info):
        raise ValueError("Baseline observed24-thread BLAS library required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "native_source_sha256": digest("casmi_ml/generated_native_scores.py"),
            "corrected_source_sha256": digest("casmi_ml/chembl_fingerprint_prior.py"),
            "config_sha256": digest(CONTROL_SOURCE / "external_config.json"),
            "sampling": "Same opaque SHA runtime-profile100 actual nonempty baseline queries per mode; no truth/outcome selection. Actual charge-aware mass window and actual encoder probabilities. Two alternating score-only repetitions for BLAS24 then1,1 then24; initialization, windows and probabilities excluded.",
            "blas_info": info,
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    encoder, saved = load_deployment_checkpoint(config["encoder"], "scale")
    encoder.eval()
    prior = np.load(config["prior"])
    index = resolve(c, config["catalog_sha256"])(pd.read_parquet(config["catalog"]))
    timing = {name: [] for name in ["blas24", "blas1"]}
    total_windows = 0
    total_candidates = 0
    max_difference = 0.0
    mismatched_windows = 0
    ranking_mismatches = 0
    for mode in ["unknown", "known"]:
        frame = pd.read_parquet(CONTROL / mode / "test.parquet")
        full = json.loads((CONTROL / mode / "baseline_full.json").read_text())
        ids = sorted(
            [r["molecule_id"] for r in full],
            key=lambda k: hashlib.sha256(("runtime-profile:" + k).encode()).digest(),
        )[:100]
        groups = frame.groupby("molecule_id", sort=False).indices
        for key in ids:
            if c.stopped():
                return None
            query = frame.iloc[groups[key]].drop(columns=["molecule_id"])
            pool, fps = candidate_window(index, query, "charge_aware_union")
            probability = group_probability(encoder, query, saved["preprocessing"])
            keys = np.asarray(pool.inchikey14)
            references = {}
            timings = {"blas24": 0.0, "blas1": 0.0}
            for threads in [24, 1, 1, 24]:
                with threadpool_limits(limits=threads, user_api="blas"):
                    started = time.perf_counter()
                    native = native_scores(probability, fps)
                    corrected = corrected_scores(probability, fps, prior)
                    timings["blas" + str(threads)] += time.perf_counter() - started
                if threads not in references:
                    references[threads] = (native, corrected)
                else:
                    if any(
                        not np.array_equal(a, b)
                        for a, b in zip(references[threads], (native, corrected))
                    ):
                        raise ValueError(
                            "Repeated same BLAS setting score values differ"
                        )
            unequal = False
            for a, b in zip(references[24], references[1]):
                unequal |= not np.array_equal(a, b)
                if len(a):
                    max_difference = max(max_difference, float(np.max(np.abs(a - b))))
                ranking_mismatches += int(
                    not np.array_equal(np.lexsort((keys, -a)), np.lexsort((keys, -b)))
                )
            mismatched_windows += int(unequal)
            total_windows += 1
            total_candidates += len(pool)
            for name, value in timings.items():
                timing[name].append(value)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "windows": total_windows,
        "candidate_rows": total_candidates,
        "score_value_mismatched_windows": mismatched_windows,
        "score_max_absolute_difference": max_difference,
        "candidate_ranking_mismatches": ranking_mismatches,
        "score_seconds_two_repetitions": {k: float(sum(v)) for k, v in timing.items()},
        "value_exact": not mismatched_windows,
        "activated": False,
        "scope": "Score-only BLAS concurrency microbenchmark; full3340 exact control required, not cold pipeline or accuracy gain",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
