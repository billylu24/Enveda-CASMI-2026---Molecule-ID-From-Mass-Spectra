"""Frozen spectrum/structure critics evaluated only on generated candidates."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from casmi_ml.chemistry import rerank
from casmi_ml.data import fingerprint, write_json
from casmi_ml.direct_models import DirectRanker, score_group
from casmi_ml.generation_slots import insert_generated
from casmi_ml.generation_temperature_experiment import BASELINE
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

SOURCE = Path("artifacts/research_loop/rounds/0005_coverage")
MODELS = {
    architecture: Path(f"artifacts/direct_rank_20260929/runs/{architecture}/model.pt")
    for architecture in ["fingerprint", "graph"]
}
SPECTRAL_COLUMNS = [
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


def critic_order(candidates, scores, weight):
    return rerank(
        [c["key"] for c in candidates],
        {},
        [],
        weight,
        top_n=max(1, len(candidates)),
        fragment_scores=scores,
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    configure(42, threads=4)
    variants = {
        "baseline": (None, 0.0),
        **{f"{a}_{w:g}": (a, w) for a in MODELS for w in [0.25, 0.5, 1.0]},
    }
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "score_source_sha256": digest(Path(score_group.__code__.co_filename)),
            "encoder_sha256": digest(ENCODER),
            "critic_sha256": {a: digest(p) for a, p in MODELS.items()},
            "generated_sha256": digest(BASELINE),
            "generated_path": str(BASELINE),
            "source_directory": str(SOURCE),
            "source_report_sha256": digest(SOURCE / "report.json"),
            "variants": variants,
            "molecules": 2000,
            "reference_guard": {"topn": 1, "threshold": 0.0},
            "open_protected": True,
            "prefix": 3,
            "slots": 5,
            "inference_device": "cpu",
            "critic_training": "Existing training-only60000 mass-neighbor contrastive checkpoints; no new fitting",
            "prior_evidence": "Graph critic failed its historical independent retrieval acceptance; this new generated-candidate study cannot override that result",
            "gan_trained": False,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "fragment_or_score_seconds": 1200,
        },
    )
    generated = json.loads(BASELINE.read_text())
    by_key = {r["key"]: r["candidates"] for r in generated}
    if len(generated) != 2000 or len(by_key) != 2000:
        raise ValueError("Full2000 generated cache required")
    cache_path = output / "scores.json"
    scores = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    frame = pq.read_table(
        ROOT / "researchdev.parquet", columns=SPECTRAL_COLUMNS, use_threads=False
    ).to_pandas()
    groups = frame.groupby("inchikey14", sort=True).indices
    encoder, saved_encoder = load_deployment_checkpoint(ENCODER, "scale")
    encoder.eval()
    rankers = {}
    for architecture, path in MODELS.items():
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if (
            checkpoint["encoder_sha256"] != digest(ENCODER)
            or checkpoint["architecture"] != architecture
        ):
            raise ValueError("Critic conditioner or architecture mismatch")
        ranker = DirectRanker(architecture).eval()
        ranker.load_state_dict(checkpoint["state_dict"])
        rankers[architecture] = ranker
    budget = StageBudget(
        output,
        "cpu_critic",
        "candidate_scoring",
        1200,
        limit=1200,
        lock_path=output / "critic.lock",
    )
    start = time.monotonic()
    try:
        for i, (key, candidates) in enumerate(by_key.items(), 1):
            if len(candidates) < 2 or key in scores:
                continue
            if not budget.checkpoint():
                raise TimeoutError("Critic scoring exceeds frozen runtime budget")
            group = frame.iloc[groups[key]].drop(columns=["inchikey14"])
            pool = pd.DataFrame(
                {"normalized_smiles": [c["smiles"] for c in candidates]}
            )
            fps = np.stack([fingerprint(c["smiles"]) for c in candidates]).astype(
                np.float32
            )
            scores[key] = {}
            for architecture, ranker in rankers.items():
                values = score_group(
                    ranker, encoder, group, saved_encoder["preprocessing"], pool, fps
                )
                if len(values) != len(candidates) or not np.isfinite(values).all():
                    raise ValueError("Invalid critic scores")
                scores[key][architecture] = {
                    c["key"]: float(value) for c, value in zip(candidates, values)
                }
            if i % 100 < 2:
                write_json(cache_path, scores)
                print("critics", i, "seconds", time.monotonic() - start, flush=True)
        write_json(cache_path, scores)
    finally:
        budget.close()
    report = {}
    for mode in ["unknown", "known"]:
        records = json.loads((SOURCE / f"{mode}_records.json").read_text())
        if {r["key"] for r in records} != set(by_key):
            raise ValueError("Generated/retrieval keys differ")
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
        for name, (architecture, weight) in variants.items():
            ranks, pools = {}, {}
            for row in records:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                allowed = protects_reference(base["ranking"], observed, confidence[key])
                selected = base if allowed else row["variants"]["expansion_1"]
                ordered = critic_order(
                    by_key[key], scores.get(key, {}).get(architecture, {}), weight
                )
                ranks[key] = (
                    insert_generated(selected["ranking"], ordered, 3, 5)
                    if allowed
                    else selected["ranking"]
                )
                pools[key] = selected["pool"] + (ordered if allowed else [])
            result, per = metrics(ranks, pools)
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostics"] = {
        "queries_scored": len(scores),
        "seconds": time.monotonic() - start,
        "new_gpu_training_seconds": 0,
        "gan_trained": False,
        "cohort": "repeated_development",
        "independent_acceptance": False,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
