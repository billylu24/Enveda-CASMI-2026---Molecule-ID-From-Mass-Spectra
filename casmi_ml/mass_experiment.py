"""Actual CPU development ranking controls for charge-aware mass hypotheses."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chemistry_experiment import chemical_ranking, reference_records
from casmi_ml.data import write_json
from casmi_ml.inference import group_probability
from casmi_ml.mass_candidates import candidate_window, hypothesis_baseline, mass_centers
from casmi_ml.metfrag import MetFrag, digest, score_group
from casmi_ml.ranking import (
    CandidateIndex,
    ReferenceIndex,
    build_candidates,
    build_reference,
    metrics,
    neural_rank,
)
from casmi_ml.research_protocol import CATALOG, ENCODER, ROOT, freeze
from casmi_ml.scale_experiment import COCONUT
from casmi_ml.secondary_inference import load_deployment_checkpoint, low_confidence_rank
from casmi_ml.training import configure


def evaluate_records(rows, variant, mode):
    chosen = [r for r in rows if mode == "unknown" or r["known"]]
    ranking = {r["key"]: r["variants"][variant]["ranking"] for r in chosen}
    pool = {r["key"]: r["variants"][variant]["pool"] for r in chosen}
    return metrics(ranking, pool)


@torch.inference_mode()
def records(output, mode, source=ROOT):
    output, source = Path(output), Path(source)
    path = output / f"{mode}_records.json"
    if path.exists():
        return json.loads(path.read_text())
    frame = pd.read_parquet(source / "researchdev.parquet")
    control = reference_records(source, "researchdev", mode)
    catalog = pd.read_parquet(CATALOG)
    allowed = (
        catalog
        if mode == "unknown"
        else catalog[
            (catalog.split == "train") | catalog.inchikey14.isin(frame.inchikey14)
        ]
    )
    # Build reference windows for all tested hypotheses, with original spectra and copy exclusions.
    refdir = output / "reference" / mode
    if not (refdir / "complete.json").exists():
        targets = set()
        for _, group in frame.groupby("inchikey14", sort=True):
            targets.update(
                mass_centers(group, "legacy")
                + mass_centers(group, "charge_aware_union")
            )
        build_reference(
            "data/train.parquet",
            allowed,
            frame,
            refdir,
            final=mode == "known",
            exclude_queries=mode == "known",
            target_centers=targets,
        )
    reference = ReferenceIndex(refdir)
    observed = set(reference.rows.inchikey14)
    if mode == "known":
        catalog = allowed[allowed.inchikey14.isin(observed)]
    index = CandidateIndex(build_candidates(catalog, COCONUT, final=mode == "known"))
    lookup = index.catalog.set_index("inchikey14").normalized_smiles.to_dict()
    groups = frame.groupby("inchikey14", sort=True).indices
    model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
    fragmenter = MetFrag(
        "external/metfrag/MetFragCommandLine-2.6.11.jar", source / "metfrag_cache"
    )
    partial = output / f"{mode}_records.partial.json"
    result = json.loads(partial.read_text()) if partial.exists() else []
    done = {r["key"] for r in result}
    frozen = {
        r["key"]: r
        for r in json.loads((source / f"researchdev_{mode}_chemical.json").read_text())
    }
    start = time.monotonic()
    for r in control:
        if r["key"] in done:
            continue
        group = frame.iloc[groups[r["key"]]]
        probability = group_probability(model, group, checkpoint["preprocessing"])
        variants = {}
        for variant in ["legacy", "charge_aware_median", "charge_aware_union"]:
            if variant == "legacy":
                # CPU neural probabilities on the original candidate/representation pool.
                keys = r["available"]["union35"]
                from casmi_ml.data import fingerprint

                fps = np.array(
                    [fingerprint(lookup[k]) for k in keys], dtype=np.float32
                ).reshape(-1, 2048)
                current = r["current_full"]
                available = list(set(keys) | set(r["available"]["coconut15"]))
            else:
                pool, fps = candidate_window(index, group, variant)
                keys = pool.inchikey14.tolist()
                current = hypothesis_baseline(group, pool, fps, reference, variant)
                available = list(set(keys) | set(r["available"]["coconut15"]))
            neural = neural_rank(probability, keys, fps)
            base = (
                r["rankings"]["coconut15"]
                if r["confidence"] >= 0.5
                else low_confidence_rank(
                    r["rankings"]["coconut15"], current, neural, 0.75
                )
            )
            fragment_scores = {}
            # Reuse identical candidate lists, otherwise compute real fragment evidence.
            if r["confidence"] < 0.5:
                if base == frozen[r["key"]]["base"]:
                    fragment_scores = frozen[r["key"]]["fragment_scores"]
                else:
                    structures = {k: lookup[k] for k in base[:100]}
                    fragment_scores, _ = score_group(
                        fragmenter, group.to_dict("records"), structures
                    )
            ranked = chemical_ranking(
                {
                    "base": base,
                    "confidence": r["confidence"],
                    "fragment_scores": fragment_scores,
                },
                "fragment",
                0.5,
            )
            variants[variant] = {
                "ranking": ranked,
                "pool": available,
                "candidate_count": len(keys),
            }
        result.append({"key": r["key"], "known": r["known"], "variants": variants})
        if len(result) % 100 == 0:
            write_json(partial, result)
            print(
                "mass ranking",
                mode,
                len(result),
                "seconds",
                time.monotonic() - start,
                flush=True,
            )
    write_json(path, result)
    return result


def run(output, source=ROOT):
    output, source = Path(output), Path(source)
    output.mkdir(parents=True, exist_ok=True)
    configure(threads=4)
    freeze(
        output / "protocol.json",
        {
            "version": 2,
            "source": str(source),
            "dev_sha256": digest(source / "researchdev.parquet"),
            "encoder_sha256": digest(ENCODER),
            "variants": ["legacy", "charge_aware_median", "charge_aware_union"],
            "historical_protection": 0.5,
            "metfrag_weight": 0.5,
            "neural_weight": 0.75,
            "path": "actual CPU group_probability; original historical retrieval; same reference exclusions",
            "molecules": 2000,
            "holdout_used": False,
        },
    )
    reports = {}
    for mode in ["unknown", "known"]:
        rows = records(output, mode, source)
        reports[mode] = {}
        for variant in ["legacy", "charge_aware_median", "charge_aware_union"]:
            report, per = evaluate_records(rows, variant, mode)
            reports[mode][variant] = report
            per.to_csv(output / f"{mode}_{variant}.csv", index=False)
    write_json(output / "report.json", reports)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--source", type=Path, default=ROOT)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.source), indent=2))


if __name__ == "__main__":
    main()
