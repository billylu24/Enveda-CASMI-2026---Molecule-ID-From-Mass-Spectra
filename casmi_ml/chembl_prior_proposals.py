"""Training-only bit-prior correction before external candidate truncation."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_fingerprint_prior import corrected_scores, marginal_prior
from casmi_ml.chembl_sequence_pilot import PROPOSALS
from casmi_ml.data import fingerprint, write_json
from casmi_ml.inference import group_probability
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics, neural_rank, rrf
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


@torch.inference_mode()
def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "prior_source_sha256": digest("casmi_ml/chembl_fingerprint_prior.py"),
            "encoder_sha256": digest(ENCODER),
            "training_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "catalog_sha256": digest(DERIVED),
            "native_proposals_sha256": digest(PROPOSALS),
            "limit": 2000,
            "candidate_limit": 500,
            "scope": "Complete frozen charge-aware mass window before top500 proposal selection; external diagnostic only",
            "variants": ["native", "corrected_fusion", "corrected_only"],
            "primary": "corrected_only",
            "prior": "Equal original60K molecule weight, Laplace(1,1) fingerprint marginal",
            "fusion": "Fixed0.5 reciprocal rank fusion on complete native and corrected rankings",
            "query_selection": "Original native proposals confidence<.5, no query keys selected using outcomes",
            "new_training": False,
            "new_sampling": False,
            "truth_used_only_in_metrics": True,
            "cohort": "repeated_development",
            "independent_acceptance": False,
            "holdout_used": False,
            "cpu_seconds": 3600,
        },
    )
    configure(42, threads=4)
    native = json.loads(PROPOSALS.read_text())
    training = pd.read_parquet(TRAIN, columns=["inchikey14", "fingerprint"])
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    if set(training.inchikey14) & set(frame.inchikey14):
        raise ValueError("Training/development overlap")
    prior = marginal_prior(training)
    del training
    np.save(output / "prior.npy", prior)
    groups = frame.groupby("inchikey14", sort=True).indices
    wanted = {k for key in groups for k in native.get(key, [])}
    catalog = pd.read_parquet(DERIVED, columns=["inchikey14", "normalized_smiles"])
    lookup = (
        catalog[catalog.inchikey14.isin(wanted)]
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    del catalog
    model, saved = load_deployment_checkpoint(ENCODER, "scale")
    model.eval()
    proposals = {name: {} for name in ("native", "corrected_fusion", "corrected_only")}
    score_count, scored_queries = 0, 0
    budget = StageBudget(
        output,
        "cpu_scoring",
        "complete_prior_proposals",
        3600,
        limit=3600,
        lock_path=output / "score.lock",
    )
    started = time.monotonic()
    try:
        for key, ids in groups.items():
            if not budget.checkpoint():
                raise TimeoutError("Full mass-window prior scoring budget exhausted")
            base = native.get(key, [])
            corrected = base
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
                fps = np.stack([fingerprint(lookup[k]) for k in base]).astype(
                    np.float32
                )
                probability = group_probability(model, query, saved["preprocessing"])
                actual_native = neural_rank(probability, base, fps)
                if actual_native != base:
                    raise ValueError(
                        "Native full mass-window ranking reconstruction differs"
                    )
                values = corrected_scores(probability, fps, prior)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite corrected proposals")
                corrected = [
                    base[i]
                    for i in sorted(
                        range(len(base)), key=lambda i: (-values[i], base[i])
                    )
                ]
                score_count += len(base)
                scored_queries += 1
                if scored_queries % 25 == 0:
                    print(
                        "full_prior_queries",
                        scored_queries,
                        "candidates",
                        score_count,
                        "seconds",
                        time.monotonic() - started,
                        flush=True,
                    )
            proposals["native"][key] = base
            proposals["corrected_only"][key] = corrected
            proposals["corrected_fusion"][key] = rrf([base, corrected], [0.5, 0.5])
        report, coverage = {}, {}
        for name, rankings in proposals.items():
            report[name], per = metrics(
                rankings, {k: v[:500] for k, v in rankings.items()}
            )
            per.to_csv(output / f"external_{name}.csv", index=False)
            coverage[name] = {
                str(n): sum(k in v[:n] for k, v in rankings.items())
                for n in (25, 100, 500)
            }
            write_json(
                output / f"{name}_proposals.json",
                {k: v for k, v in rankings.items() if k in native},
            )
        result = {
            "diagnostic_only": True,
            "external_only": report,
            "coverage_counts": coverage,
            "scored_queries": scored_queries,
            "scored_candidates": score_count,
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "prior_sha256": digest(output / "prior.npy"),
            "independent_acceptance": False,
            "next_gate": "Proposal coverage/ranking cannot qualify release; full combined0062 and known protection required",
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
