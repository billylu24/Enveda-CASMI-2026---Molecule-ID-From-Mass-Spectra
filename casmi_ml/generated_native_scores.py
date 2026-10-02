"""Frozen native fingerprint evidence on current calibrated decoder candidates."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.chemistry import rerank
from casmi_ml.data import fingerprint, write_json
from casmi_ml.generated_score_combination import (
    GENERATED,
    SCORES,
    SOURCE,
    combined_order,
)
from casmi_ml.generated_second_reference import select_prefix
from casmi_ml.generation_slots import insert_generated
from casmi_ml.inference import group_probability
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

VARIANTS = {
    "baseline": (True, 0),
    "native025": (False, 0.25),
    "native05": (False, 0.5),
    "native1": (False, 1),
    "critic_native025": (True, 0.25),
    "critic_native05": (True, 0.5),
}


def native_scores(probability, fps):
    p = np.clip(np.asarray(probability, dtype=float), 1e-6, 1 - 1e-6)
    fps = np.asarray(fps, dtype=float)
    if (
        p.shape != (2048,)
        or fps.ndim != 2
        or fps.shape[1] != 2048
        or not np.isfinite(p).all()
        or not np.isfinite(fps).all()
    ):
        raise ValueError("Fingerprint score dimensions/values invalid")
    return fps @ (np.log(p) - np.log1p(-p)) / 2048


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
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": VARIANTS,
            "ranking": "Measured token exponent1; optional existing critic.5; native Bernoulli fingerprint ranking evidence",
            "routing": "0047 adaptive original/expanded;5 slots;128 fixed trajectories",
            "new_training": False,
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "cohort": "repeated_development",
        },
    )
    generated = json.loads(GENERATED.read_text())
    candidates = {r["key"]: r["candidates"] for r in generated}
    if len(generated) != 2000 or len(candidates) != 2000:
        raise ValueError("Require2000 complete")
    columns = [
        "inchikey14",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "adduct",
        "instrument_type",
        "ionization_mode",
        "collision_energy_ev",
        "precursor_error_ppm",
    ]
    frame = pd.read_parquet(ROOT / "researchdev.parquet", columns=columns)
    groups = frame.groupby("inchikey14", sort=True).indices
    encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
    encoder.eval()
    critic = json.loads(SCORES.read_text())
    scores = {}
    budget = StageBudget(
        output,
        "cpu_native",
        "score",
        1200,
        limit=1200,
        lock_path=output / "native.lock",
    )
    started = time.monotonic()
    try:
        for key, values in candidates.items():
            if len(values) < 2:
                continue
            if not budget.checkpoint():
                raise TimeoutError("Native score budget exhausted")
            probability = group_probability(
                encoder,
                frame.iloc[groups[key]].drop(columns=["inchikey14"]),
                saved["preprocessing"],
            )
            fps = np.stack([fingerprint(c["smiles"]) for c in values])
            scores[key] = {
                c["key"]: float(v)
                for c, v in zip(values, native_scores(probability, fps))
            }
    finally:
        budget.close()
    write_json(output / "scores.json", scores)
    orderings = {}
    for name, (use_critic, weight) in VARIANTS.items():
        orderings[name] = {}
        for key, values in candidates.items():
            base = combined_order(
                values,
                critic.get(key, {}).get("fingerprint", {}),
                (1, 0, 0.5 if use_critic else 0),
            )
            orderings[name][key] = rerank(
                base,
                {},
                [],
                weight,
                top_n=max(1, len(base)),
                fragment_scores=scores.get(key, {}),
            )
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
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostics"] = {
        "queries_scored": len(scores),
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
