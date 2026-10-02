"""Query-observable critic margin for promoting the first novel generated candidate."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import fingerprint, write_json
from casmi_ml.direct_models import DirectRanker, score_group
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
VARIANTS = {"baseline": None, "margin0": 0.0, "margin005": 0.05, "margin01": 0.1}


def promotion_prefix(ranking, generated, scores, prefix, margin):
    if margin is None or not ranking:
        return prefix
    novel = [key for key in generated if key not in set(ranking)]
    if not novel or ranking[0] not in scores or novel[0] not in scores:
        return prefix
    return 0 if scores[novel[0]] > scores[ranking[0]] + margin else prefix


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
            "calibration_scores_sha256": digest(SCORES),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": VARIANTS,
            "criterion": "First novel calibrated generated candidate critic cosine exceeds first actual retrieval candidate by margin; else0047 prefix",
            "routing": "0047 original/expanded;5 slots;128 fixed trajectories",
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "cohort": "repeated_development",
            "new_training": False,
        },
    )
    generated = json.loads(GENERATED.read_text())
    candidates = {r["key"]: r["candidates"] for r in generated}
    old_scores = json.loads(SCORES.read_text())
    ordered = {
        k: combined_order(v, old_scores.get(k, {}).get("fingerprint", {}), (1, 0, 0.5))
        for k, v in candidates.items()
    }
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
    encoder.eval()
    weights = torch.load(CRITIC, map_location="cpu", weights_only=True)
    assert (
        weights["encoder_sha256"] == digest(ENCODER)
        and weights["architecture"] == "fingerprint"
    )
    ranker = DirectRanker("fingerprint").eval()
    ranker.load_state_dict(weights["state_dict"])
    from casmi_ml.chemistry_experiment import candidate_lookup

    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    extra = pd.read_parquet("external/pubchemlite/structures.parquet")
    lookup.update(
        {
            r.inchikey14: r.normalized_smiles
            for r in extra.itertuples()
            if r.inchikey14 not in lookup
        }
    )
    report = {}
    score_cache = {}
    diagnostics = {}
    budget = StageBudget(
        output,
        "cpu_margin",
        "score",
        1200,
        limit=1200,
        lock_path=output / "margin.lock",
    )
    try:
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
            if mode == "known":
                lookup.update(candidate_lookup(ROOT, "researchdev", "known"))
            selected_rows = []
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                protected = protects_reference(
                    base["ranking"], observed, confidence[key]
                )
                selected = base if protected else row["variants"]["expansion_1"]
                ranking = selected["ranking"]
                novel = [k for k in ordered[key] if k not in set(ranking)]
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
                scores = {}
                if ranking and novel:
                    if not budget.checkpoint():
                        raise TimeoutError("Margin score budget exhausted")
                    cache_key = key + ":" + ranking[0] + ":" + novel[0]
                    if cache_key not in score_cache:
                        generated_lookup = {
                            c["key"]: c["smiles"] for c in candidates[key]
                        }
                        pair = [lookup[ranking[0]], generated_lookup[novel[0]]]
                        fps = np.stack([fingerprint(s) for s in pair]).astype(
                            np.float32
                        )
                        group = frame.iloc[groups[key]].drop(
                            columns=[
                                c
                                for c in [
                                    "inchikey14",
                                    "normalized_smiles",
                                    "molecular_formula",
                                    "fingerprint",
                                ]
                                if c in frame
                            ]
                        )
                        values = score_group(
                            ranker,
                            encoder,
                            group,
                            saved["preprocessing"],
                            pd.DataFrame({"normalized_smiles": pair}),
                            fps,
                        )
                        score_cache[cache_key] = {
                            ranking[0]: float(values[0]),
                            novel[0]: float(values[1]),
                        }
                    scores = score_cache[cache_key]
                selected_rows.append((key, selected, prefix, scores))
            report[mode] = {}
            diagnostics[mode] = {}
            for name, margin in VARIANTS.items():
                ranks, pools = {}, {}
                promoted = 0
                for key, selected, prefix, scores in selected_rows:
                    effective = promotion_prefix(
                        selected["ranking"], ordered[key], scores, prefix, margin
                    )
                    promoted += effective == 0
                    ranks[key] = insert_generated(
                        selected["ranking"], ordered[key], effective, 5
                    )
                    pools[key] = selected["pool"] + ordered[key]
                result, per = metrics(ranks, pools)
                report[mode][name] = result
                diagnostics[mode][name] = {"queries_with_promotion": promoted}
                per.to_csv(output / f"{mode}_{name}.csv", index=False)
    finally:
        budget.close()
    write_json(output / "scores.json", score_cache)
    write_json(output / "diagnostics.json", diagnostics)
    report["diagnostics"] = diagnostics
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
