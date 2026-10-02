"""Frozen spectral fingerprint ranking of already mass-matched generated structures."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.data import fingerprint, write_json
from casmi_ml.generation_slots import insert_generated
from casmi_ml.inference import group_probability
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics, neural_rank, rrf
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint


def rank_generated(candidates, probability, weight):
    if weight < 0 or not np.isfinite(weight):
        raise ValueError("Finite nonnegative generated neural weight required")
    original = [c["key"] for c in candidates]
    if len(original) < 2 or not weight:
        return original
    fps = np.asarray([fingerprint(c["smiles"]) for c in candidates], dtype=np.float32)
    return rrf([original, neural_rank(probability, original, fps)], [1.0, weight])


def run(output, source=Path("artifacts/research_loop/rounds/0005_coverage")):
    output, source = Path(output), Path(source)
    output.mkdir(parents=True, exist_ok=True)
    generated_path = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
    generated = {
        r["key"]: r["candidates"] for r in json.loads(generated_path.read_text())
    }
    variants = {
        "baseline": 0.0,
        "neural_0.25": 0.25,
        "neural_0.5": 0.5,
        "neural_1": 1.0,
    }
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_directory": str(source),
            "source_report_sha256": digest(source / "report.json"),
            "generated_sha256": digest(generated_path),
            "encoder_sha256": digest(ENCODER),
            "variants": variants,
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "prefix": 5,
            "slots": 5,
            "probability_path": "CPU group_probability identical to retrieval",
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
        },
    )
    scores_path = output / "probabilities.json"
    probabilities = json.loads(scores_path.read_text()) if scores_path.exists() else {}
    if not scores_path.exists():
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        groups = frame.groupby("inchikey14", sort=True).indices
        model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
        for i, (key, candidates) in enumerate(generated.items(), 1):
            if len(candidates) >= 2:
                group = frame.iloc[groups[key]]
                probabilities[key] = group_probability(
                    model, group, checkpoint["preprocessing"]
                ).tolist()
            if i % 200 == 0:
                print("probabilities", i, flush=True)
        write_json(scores_path, probabilities)
    reordered = {
        name: {
            key: rank_generated(
                candidates, np.asarray(probabilities.get(key, [])), weight
            )
            for key, candidates in generated.items()
        }
        for name, weight in variants.items()
    }
    report = {}
    for mode in ["unknown", "known"]:
        rows = json.loads((source / f"{mode}_records.json").read_text())
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
        for name in variants:
            ranks, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key = row["key"]
                base = row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                ranks[key] = (
                    insert_generated(selected["ranking"], reordered[name][key], 5, 5)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (
                    [c["key"] for c in generated[key]] if allowed else []
                )
            m, per = metrics(ranks, pools)
            report[mode][name] = m
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))


if __name__ == "__main__":
    main()
