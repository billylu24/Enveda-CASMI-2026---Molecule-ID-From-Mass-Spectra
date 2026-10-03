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
from casmi_ml.mass_candidates import candidate_window
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex, metrics, neural_rank, rrf
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def ordered_proposals(keys, values):
    if len(keys) != len(values) or not np.isfinite(values).all():
        raise ValueError("Proposal identities and finite scores must align")
    return [
        key
        for key, _ in sorted(zip(keys, values), key=lambda pair: (-pair[1], pair[0]))
    ]


@torch.inference_mode()
def run(output, all_queries=False, encoder=ENCODER):
    encoder = Path(encoder)
    alternate_encoder = encoder.resolve() != Path(ENCODER).resolve()
    if alternate_encoder and not all_queries:
        raise ValueError("Alternate encoder requires complete all-query mass windows")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "prior_source_sha256": digest("casmi_ml/chembl_fingerprint_prior.py"),
            "encoder_sha256": digest(encoder),
            "encoder_path": str(encoder),
            "alternate_encoder": alternate_encoder,
            "training_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "catalog_sha256": digest(DERIVED),
            "native_proposals_sha256": digest(PROPOSALS),
            "limit": 2000,
            "candidate_limit": 500,
            "all_queries": all_queries,
            "scope": "Complete frozen charge-aware mass window before top500 proposal selection; external diagnostic only",
            "variants": ["native", "corrected_fusion", "corrected_only"],
            "primary": "corrected_only",
            "prior": "Equal original60K molecule weight, Laplace(1,1) fingerprint marginal",
            "fusion": "Fixed0.5 reciprocal rank fusion on complete native and corrected rankings",
            "query_selection": "All development query keys"
            if all_queries
            else "Original native proposals confidence<.5; outcomes never select query keys",
            "native_reconstruction": "Original encoder requires exact stored native rank; alternate encoder preserves exact frozen0147 all-query mass pool, with new scores in stable mass/key window order",
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
    frozen_all_native = (
        json.loads(
            Path(
                "artifacts/research_loop/rounds/0147_chembl_all_query_proposals/native_proposals.json"
            ).read_text()
        )
        if alternate_encoder
        else None
    )
    training = pd.read_parquet(TRAIN, columns=["inchikey14", "fingerprint"])
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    if set(training.inchikey14) & set(frame.inchikey14):
        raise ValueError("Training/development overlap")
    prior = marginal_prior(training)
    del training
    np.save(output / "prior.npy", prior)
    groups = frame.groupby("inchikey14", sort=True).indices
    wanted = {k for key in groups for k in native.get(key, [])}
    catalog = pd.read_parquet(
        DERIVED, columns=["inchikey14", "normalized_smiles", "mass"]
    )
    index = CandidateIndex(catalog) if all_queries else None
    wanted_catalog = catalog[catalog.inchikey14.isin(wanted)].set_index("inchikey14")
    lookup = wanted_catalog.normalized_smiles.to_dict()
    mass_lookup = wanted_catalog.mass.to_dict()
    del wanted_catalog
    del catalog
    model, saved = load_deployment_checkpoint(encoder, "scale")
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
            if len(base) >= 2 or (all_queries and key not in native):
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
                if all_queries and key not in native:
                    if len(index.cache) > 20000:
                        index.cache.clear()
                    pool, fps = candidate_window(index, query, "charge_aware_union")
                    scoring_order = pool.inchikey14.tolist()
                else:
                    scoring_order = sorted(base, key=lambda k: (mass_lookup[k], k))
                    fps = np.stack(
                        [fingerprint(lookup[k]) for k in scoring_order]
                    ).astype(np.float32)
                if not scoring_order:
                    for rankings in proposals.values():
                        rankings[key] = []
                    continue
                probability = group_probability(model, query, saved["preprocessing"])
                actual_native = neural_rank(probability, scoring_order, fps)
                if alternate_encoder and set(scoring_order) != set(
                    frozen_all_native[key]
                ):
                    raise ValueError(
                        "Alternate encoder changed frozen observable mass pool"
                    )
                if not alternate_encoder and key in native and actual_native != base:
                    raise ValueError(
                        "Native full mass-window ranking reconstruction differs"
                    )
                if alternate_encoder or key not in native:
                    base = actual_native
                values = corrected_scores(probability, fps, prior)
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite corrected proposals")
                corrected = ordered_proposals(scoring_order, values)
                score_count += len(scoring_order)
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
                rankings
                if all_queries
                else {k: v for k, v in rankings.items() if k in native},
            )
        result = {
            "diagnostic_only": True,
            "external_only": report,
            "coverage_counts": coverage,
            "all_queries": all_queries,
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
    parser.add_argument("--all-queries", action="store_true")
    parser.add_argument("--encoder", type=Path, default=ENCODER)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.all_queries, args.encoder), indent=2))


if __name__ == "__main__":
    main()
