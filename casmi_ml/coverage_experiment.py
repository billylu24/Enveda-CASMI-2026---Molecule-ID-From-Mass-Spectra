"""Public catalog expansion with frozen mass, fingerprint and real fragmentation ranks."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch

from casmi_ml.candidate_catalog import expanded_pool
from casmi_ml.chemistry_experiment import chemical_ranking, reference_records
from casmi_ml.data import write_json
from casmi_ml.inference import group_probability
from casmi_ml.mass_candidates import candidate_window, hypothesis_baseline
from casmi_ml.metfrag import MetFrag, digest, score_group
from casmi_ml.ranking import (
    CandidateIndex,
    ReferenceIndex,
    build_candidates,
    metrics,
    neural_rank,
    rrf,
)
from casmi_ml.research_protocol import CATALOG, ENCODER, ROOT, freeze
from casmi_ml.scale_experiment import COCONUT
from casmi_ml.secondary_inference import load_deployment_checkpoint, low_confidence_rank
from casmi_ml.training import configure

EXTERNAL = Path("external/pubchemlite/structures.parquet")


@torch.inference_mode()
def records(output, incumbent, mode):
    path = output / f"{mode}_records.json"
    if path.exists():
        return json.loads(path.read_text())
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    controls = {r["key"]: r for r in reference_records(ROOT, "researchdev", mode)}
    old = json.loads((incumbent / f"{mode}_records.json").read_text())
    reference = ReferenceIndex(incumbent / "reference" / mode)
    catalog = pd.read_parquet(CATALOG)
    if mode == "known":
        catalog = catalog[
            ((catalog.split == "train") | catalog.inchikey14.isin(frame.inchikey14))
            & catalog.inchikey14.isin(reference.rows.inchikey14)
        ]
    original = build_candidates(catalog, COCONUT, final=mode == "known")
    external = pd.read_parquet(EXTERNAL)
    index = CandidateIndex(expanded_pool(original, external))
    lookup = index.catalog.set_index("inchikey14").normalized_smiles.to_dict()
    model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
    groups = frame.groupby("inchikey14", sort=True).indices
    fragmenter = MetFrag(
        "external/metfrag/MetFragCommandLine-2.6.11.jar", ROOT / "metfrag_cache"
    )
    partial = output / f"{mode}_records.partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    done = {r["key"] for r in rows}
    start = time.monotonic()
    for row in old:
        key = row["key"]
        if key in done:
            continue
        control = controls[key]
        base = row["variants"]["charge_aware_union"]
        group = frame.iloc[groups[key]]
        pool, fps = candidate_window(index, group, "charge_aware_union")
        ranking = base["ranking"]
        if control["confidence"] < 0.5:
            probability = group_probability(model, group, checkpoint["preprocessing"])
            current = hypothesis_baseline(
                group, pool, fps, reference, "charge_aware_union"
            )
            neural = neural_rank(probability, pool.inchikey14.tolist(), fps)
            raw = low_confidence_rank(
                control["rankings"]["coconut15"], current, neural, 0.75
            )
            scores, _ = score_group(
                fragmenter, group.to_dict("records"), {k: lookup[k] for k in raw[:100]}
            )
            ranking = chemical_ranking(
                {
                    "base": raw,
                    "confidence": control["confidence"],
                    "fragment_scores": scores,
                },
                "fragment",
                0.5,
            )
        available = list(set(pool.inchikey14) | set(control["available"]["coconut15"]))
        variants = {"baseline": base}
        for weight in [0.25, 0.5, 1.0]:
            variants[f"expansion_{weight:g}"] = {
                "ranking": base["ranking"]
                if control["confidence"] >= 0.5
                else rrf([base["ranking"], ranking], [1 - weight, weight]),
                "pool": available,
                "candidate_count": len(pool),
            }
        rows.append({"key": key, "known": row["known"], "variants": variants})
        if len(rows) % 100 == 0:
            write_json(partial, rows)
            print(
                "coverage ranking",
                mode,
                len(rows),
                "seconds",
                time.monotonic() - start,
                flush=True,
            )
    write_json(path, rows)
    return rows


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    configure(threads=4)
    provenance = json.loads(Path("external/pubchemlite/manifest.json").read_text())
    if digest(EXTERNAL) != provenance["derived_sha256"]:
        raise ValueError("External catalog checksum changed")
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "external": provenance,
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "encoder_sha256": digest(ENCODER),
            "incumbent_directory": str(incumbent),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "weights": [0.25, 0.5, 1.0],
            "mass_hypothesis": "charge_aware_union",
            "metfrag_weight": 0.5,
            "neural_weight": 0.75,
            "holdout_used": False,
            "labels_used_for_candidate_selection": False,
        },
    )
    report = {}
    for mode in ["unknown", "known"]:
        rows = records(output, incumbent, mode)
        chosen = [r for r in rows if mode == "unknown" or r["known"]]
        report[mode] = {}
        for variant in ["baseline", "expansion_0.25", "expansion_0.5", "expansion_1"]:
            ranks = {r["key"]: r["variants"][variant]["ranking"] for r in chosen}
            pools = {r["key"]: r["variants"][variant]["pool"] for r in chosen}
            result, per = metrics(ranks, pools)
            report[mode][variant] = result
            per.to_csv(output / f"{mode}_{variant}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.incumbent), indent=2))


if __name__ == "__main__":
    main()
