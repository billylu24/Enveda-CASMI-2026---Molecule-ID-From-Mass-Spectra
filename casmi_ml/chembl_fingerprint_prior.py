"""Training-only fingerprint marginal correction of external candidate evidence."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_sequence_pilot import PROPOSALS
from casmi_ml.chemistry import rerank
from casmi_ml.data import fingerprint, write_json
from casmi_ml.generated_native_scores import native_scores
from casmi_ml.inference import group_probability
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def marginal_prior(training):
    rows = training.drop_duplicates("inchikey14", keep="first")
    values = np.stack(
        [np.unpackbits(np.frombuffer(v, dtype=np.uint8)) for v in rows.fingerprint]
    )
    if values.shape != (60000, 2048):
        raise ValueError("Require exactly60K original training molecule fingerprints")
    return (values.sum(0, dtype=np.float64) + 1) / (len(rows) + 2)


def corrected_scores(probability, fps, prior):
    prior = np.asarray(prior, dtype=float)
    if (
        prior.shape != (2048,)
        or not np.isfinite(prior).all()
        or ((prior <= 0) | (prior >= 1)).any()
    ):
        raise ValueError("Finite positive interior training marginals required")
    return (
        native_scores(probability, fps)
        - fps @ (np.log(prior) - np.log1p(-prior)) / 2048
    )


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "encoder_sha256": digest(ENCODER),
            "training_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "catalog_sha256": digest(DERIVED),
            "proposals_sha256": digest(PROPOSALS),
            "limit": 2000,
            "candidate_limit": 500,
            "prior": "Equal molecule weight on original60K training fingerprints; Laplace(1,1) bit marginals",
            "score": "Mean per-bit Bernoulli log likelihood ratio between actual predicted p and training marginal q",
            "variants": {
                "native": None,
                "corrected_fusion": 0.5,
                "corrected_only": 1.0,
            },
            "new_training": False,
            "new_sampling": False,
            "truth_used_only_in_metrics": True,
            "cohort": "repeated_development",
            "independent_acceptance": False,
            "scope": "External ranking diagnostic only; not combined ranking release eligibility",
        },
    )
    configure(42, threads=4)
    training = pd.read_parquet(TRAIN, columns=["inchikey14", "fingerprint"])
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    if set(training.inchikey14) & set(frame.inchikey14):
        raise ValueError("Training/development overlap")
    prior = marginal_prior(training)
    del training
    np.save(output / "prior.npy", prior)
    proposals = json.loads(PROPOSALS.read_text())
    groups = frame.groupby("inchikey14", sort=True).indices
    wanted = {k for key in groups for k in proposals.get(key, [])[:500]}
    catalog = pd.read_parquet(DERIVED, columns=["inchikey14", "normalized_smiles"])
    lookup = (
        catalog.loc[catalog.inchikey14.isin(wanted)]
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    del catalog
    model, saved = load_deployment_checkpoint(ENCODER, "scale")
    model.eval()
    ranks = {name: {} for name in ("native", "corrected_fusion", "corrected_only")}
    pools, cache = {}, {}
    budget = StageBudget(
        output,
        "cpu_scoring",
        "fingerprint_prior",
        3600,
        limit=3600,
        lock_path=output / "score.lock",
    )
    started = time.monotonic()
    try:
        with torch.inference_mode():
            for key, ids in groups.items():
                if not budget.checkpoint():
                    raise TimeoutError("Fingerprint prior diagnostic budget exhausted")
                base = proposals.get(key, [])[:500]
                pools[key] = base
                ranks["native"][key] = base
                values = {}
                if len(base) >= 2:
                    query = frame.iloc[ids].drop(
                        columns=[
                            c
                            for c in (
                                "inchikey14",
                                "normalized_smiles",
                                "fingerprint",
                                "molecular_formula",
                            )
                            if c in frame
                        ]
                    )
                    probability = group_probability(
                        model, query, saved["preprocessing"]
                    )
                    fps = np.stack([fingerprint(lookup[k]) for k in base])
                    scores = corrected_scores(probability, fps, prior)
                    if not np.isfinite(scores).all():
                        raise ValueError("Nonfinite marginal-corrected scores")
                    values = {k: float(v) for k, v in zip(base, scores)}
                    cache[key] = values
                for name, weight in (
                    ("corrected_fusion", 0.5),
                    ("corrected_only", 1.0),
                ):
                    ranks[name][key] = rerank(
                        base,
                        {},
                        [],
                        weight,
                        top_n=max(1, len(base)),
                        fragment_scores=values,
                    )
        write_json(output / "scores.json", cache)
        report = {}
        for name, ranking in ranks.items():
            report[name], per = metrics(ranking, pools)
            per.to_csv(output / f"external_{name}.csv", index=False)
        result = {
            "diagnostic_only": True,
            "external_only": report,
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "scored_queries": len(cache),
            "prior_sha256": digest(output / "prior.npy"),
            "new_training": False,
            "independent_acceptance": False,
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
