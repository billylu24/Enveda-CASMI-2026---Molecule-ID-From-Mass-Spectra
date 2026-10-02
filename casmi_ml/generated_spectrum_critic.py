"""Matched multispectrum critic aggregation with fixed decoder and routing."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import features, fingerprint, write_json
from casmi_ml.direct_models import DirectRanker
from casmi_ml.generated_score_combination import (
    GENERATED,
    SCORES,
    SOURCE,
    combined_order,
)
from casmi_ml.generated_second_reference import select_prefix
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

CRITIC = Path("artifacts/direct_rank_20260929/runs/fingerprint/model.pt")
VARIANTS = ["baseline", "max", "median", "top2"]


def aggregate(scores, variant):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 2 or not len(scores) or not np.isfinite(scores).all():
        raise ValueError("Require finite spectrum-by-candidate scores")
    if variant == "baseline":
        return scores.mean(0)
    if variant == "max":
        return scores.max(0)
    if variant == "median":
        return np.median(scores, axis=0)
    if variant == "top2":
        return np.sort(scores, axis=0)[-2:].mean(0)
    raise ValueError("Unknown critic aggregation")


@torch.inference_mode()
def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    configure(42, threads=4)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "encoder_sha256": digest(ENCODER),
            "critic_sha256": digest(CRITIC),
            "generated_sha256": digest(GENERATED),
            "baseline_scores_sha256": digest(SCORES),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": VARIANTS,
            "critic_weight": 0.5,
            "token_length_exponent": 1,
            "routing": "0047 adaptive baseline/expanded;5 slots;128 frozen trajectories",
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "cohort": "repeated_development",
            "new_training": False,
            "score_device": "cpu",
            "score_seconds": 1200,
        },
    )
    generated = json.loads(GENERATED.read_text())
    candidates = {r["key"]: r["candidates"] for r in generated}
    if len(generated) != 2000 or len(candidates) != 2000:
        raise ValueError("Require full2000")
    frame = pd.read_parquet(
        ROOT / "researchdev.parquet",
        columns=[
            "inchikey14",
            "ms2_mzs",
            "ms2_normalized_intensities",
            "precursor_mz",
            "adduct",
            "instrument_type",
            "ionization_mode",
            "collision_energy_ev",
            "precursor_error_ppm",
        ],
    )
    groups = frame.groupby("inchikey14", sort=True).indices
    encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
    encoder.eval()
    weights = torch.load(CRITIC, map_location="cpu", weights_only=True)
    if weights["architecture"] != "fingerprint" or weights["encoder_sha256"] != digest(
        ENCODER
    ):
        raise ValueError("Frozen critic mismatch")
    ranker = DirectRanker("fingerprint").eval()
    ranker.load_state_dict(weights["state_dict"])
    reference = json.loads(SCORES.read_text())
    results = {}
    budget = StageBudget(
        output,
        "cpu_critic",
        "multispectrum",
        1200,
        limit=1200,
        lock_path=output / "critic.lock",
    )
    started = time.monotonic()
    try:
        for key, values in candidates.items():
            if len(values) < 2:
                continue
            if not budget.checkpoint():
                raise TimeoutError("Critic score budget exhausted")
            group = frame.iloc[groups[key]].drop(columns=["inchikey14"])
            rows = [
                features(r, saved["preprocessing"]) for r in group.to_dict("records")
            ]
            pieces = [
                torch.from_numpy(np.stack([r[i] for r in rows])) for i in (0, 2, 1)
            ]
            latent = encoder.encoder(torch.cat(pieces, -1))
            queries = ranker.encode_spectra(latent)
            fps = torch.from_numpy(
                np.stack([fingerprint(c["smiles"]) for c in values]).astype(np.float32)
            )
            structures = ranker.encode_molecules(fps)
            matrix = (queries @ structures.T).numpy()
            expected = reference[key]["fingerprint"]
            actual = aggregate(matrix, "baseline")
            if (
                max(abs(actual[i] - expected[c["key"]]) for i, c in enumerate(values))
                > 1e-6
            ):
                raise ValueError("Mean critic scores differ from frozen baseline")
            results[key] = {
                name: {
                    c["key"]: float(value)
                    for c, value in zip(values, aggregate(matrix, name))
                }
                for name in VARIANTS
            }
    finally:
        budget.close()
    write_json(output / "scores.json", results)
    orderings = {
        name: {
            key: combined_order(values, results.get(key, {}).get(name, {}), (1, 0, 0.5))
            for key, values in candidates.items()
        }
        for name in VARIANTS
    }
    report = {}
    for mode in ("unknown", "known"):
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        observed = set(
            pd.read_parquet(
                Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                / mode
                / "rows.parquet",
                columns=["inchikey14"],
            ).inchikey14
        )
        report[mode] = {}
        for name in VARIANTS:
            ranks, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                protected = protects_reference(
                    base["ranking"], observed, confidence[key]
                )
                selected = base if protected else row["variants"]["expansion_1"]
                prefix = (
                    select_prefix(
                        base["ranking"],
                        observed,
                        confidence[key],
                        "second_unreferenced",
                    )
                    if protected
                    else 2
                )
                ranks[key] = insert_generated(
                    selected["ranking"], orderings[name][key], prefix, 5
                )
                pools[key] = selected["pool"] + orderings[name][key]
            result, per = metrics(ranks, pools)
            if name == "baseline":
                expected = (
                    pd.read_csv(incumbent / f"{mode}_expanded_prefix2.csv")
                    .set_index("key")
                    .sort_index()
                )
                actual = per.set_index("key").sort_index()
                if (
                    set(actual.index) != set(expected.index)
                    or (
                        actual[["reciprocal_rank", "top1"]]
                        - expected[["reciprocal_rank", "top1"]]
                    )
                    .abs()
                    .max()
                    .max()
                    > 1e-12
                ):
                    raise ValueError("Paired baseline must match0047")
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostics"] = {
        "candidate_queries_scored": len(results),
        "mean_scores_match_frozen_baseline": True,
        "elapsed_seconds": time.monotonic() - started,
        "independent_acceptance": False,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.incumbent), indent=2))


if __name__ == "__main__":
    main()
