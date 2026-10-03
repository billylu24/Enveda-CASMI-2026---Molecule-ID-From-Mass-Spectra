"""Mass-conditioned training-only fingerprint background before truncation."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from baseline import formula_mass
from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_fingerprint_prior import corrected_scores
from casmi_ml.chembl_sequence_pilot import PROPOSALS
from casmi_ml.data import fingerprint, write_json
from casmi_ml.inference import group_probability
from casmi_ml.mass_candidates import candidate_window, mass_centers
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex, metrics, neural_rank, rrf
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def local_prior(center, masses, keys, fps, neighbors=1024):
    if (
        not np.isfinite(center)
        or len(masses) < neighbors
        or len(keys) != len(masses)
        or len(fps) != len(masses)
    ):
        raise ValueError(
            "Finite observable mass and enough aligned training rows required"
        )
    pos = int(np.searchsorted(masses, center))
    lo, hi = max(0, pos - neighbors), min(len(masses), pos + neighbors)
    distances = np.abs(masses[lo:hi] - center)
    radius = np.partition(distances, neighbors - 1)[neighbors - 1]
    # Include every boundary tie before resolving by key; a duplicate mass run
    # can extend beyond the initial index neighborhood.
    lo, hi = np.searchsorted(masses, [center - radius, center + radius], side="left")
    hi = np.searchsorted(masses, center + radius, side="right")
    order = np.lexsort((keys[lo:hi], np.abs(masses[lo:hi] - center)))[:neighbors] + lo
    return (fps[order].sum(0, dtype=np.float64) + 1) / (neighbors + 2)


def ordered_proposals(keys, values):
    if len(keys) != len(values) or not np.isfinite(values).all():
        raise ValueError("Proposal identities and finite scores must align")
    return [
        key
        for key, _ in sorted(zip(keys, values), key=lambda pair: (-pair[1], pair[0]))
    ]


@torch.inference_mode()
def run(output, all_queries=True):
    if not all_queries:
        raise ValueError("All queries required for this fixed diagnostic")
    output = Path(output)
    control = Path("artifacts/research_loop/rounds/0147_chembl_all_query_proposals")
    global_prior_path = Path(
        "artifacts/research_loop/rounds/0137_chembl_full_prior_proposals/prior.npy"
    )
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "prior_source_sha256": digest("casmi_ml/chembl_fingerprint_prior.py"),
            "background_neighbors": 1024,
            "matched_control_protocol_sha256": digest(control / "protocol.json"),
            "matched_control_native_sha256": digest(control / "native_proposals.json"),
            "matched_control_corrected_sha256": digest(
                control / "corrected_only_proposals.json"
            ),
            "global_prior_sha256": digest(global_prior_path),
            "encoder_sha256": digest(ENCODER),
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
            "prior": "Nearest1024 original60K molecules to median observable charge-aware neutral mass; distance/key stable order; Laplace(1,1) bit marginals. No development labels or candidate identity used in background selection",
            "fusion": "Fixed0.5 reciprocal rank fusion on complete native and corrected rankings",
            "query_selection": "All development query keys"
            if all_queries
            else "Original native proposals confidence<.5; outcomes never select query keys",
            "native_reconstruction": "Compute fingerprint matrix in original stable mass/key window order to preserve BLAS numerical arithmetic; require exact stored native rank",
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
    control_native = json.loads((control / "native_proposals.json").read_text())
    control_corrected = json.loads(
        (control / "corrected_only_proposals.json").read_text()
    )
    global_prior = np.load(global_prior_path)
    training = pd.read_parquet(
        TRAIN, columns=["inchikey14", "fingerprint", "molecular_formula"]
    ).drop_duplicates("inchikey14")
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    if set(training.inchikey14) & set(frame.inchikey14):
        raise ValueError("Training/development overlap")
    if len(training) != 60000:
        raise ValueError("Exactly original60K training molecules required")
    training_masses = training.molecular_formula.map(formula_mass).to_numpy(dtype=float)
    training_keys = training.inchikey14.to_numpy()
    training_fps = np.stack(
        [np.unpackbits(np.frombuffer(v, dtype=np.uint8)) for v in training.fingerprint]
    )
    if training_fps.shape != (60000, 2048) or not np.isfinite(training_masses).all():
        raise ValueError("Invalid training fingerprints or masses")
    training_order = np.lexsort((training_keys, training_masses))
    training_masses = training_masses[training_order]
    training_keys = training_keys[training_order]
    training_fps = training_fps[training_order]
    del training
    np.savez(
        output / "training_background.npz",
        masses=training_masses,
        keys=training_keys,
        fps=training_fps,
    )
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
                if key in native and actual_native != base:
                    raise ValueError(
                        "Native full mass-window ranking reconstruction differs"
                    )
                if actual_native != control_native[key]:
                    raise ValueError("Matched full native control differs")
                actual_global = ordered_proposals(
                    scoring_order, corrected_scores(probability, fps, global_prior)
                )
                if actual_global != control_corrected[key]:
                    raise ValueError("Matched full global prior control differs")
                if key not in native:
                    base = actual_native
                centers = mass_centers(query, "charge_aware_median")
                if not centers:
                    raise ValueError("Observable mass center missing for nonempty pool")
                prior = local_prior(
                    centers[0], training_masses, training_keys, training_fps
                )
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
            "matched_control_metrics": json.loads(
                (control / "report.json").read_text()
            )["external_only"],
            "matched_native_and_global_reconstruction": True,
            "coverage_counts": coverage,
            "all_queries": all_queries,
            "scored_queries": scored_queries,
            "scored_candidates": score_count,
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "training_background_sha256": digest(output / "training_background.npz"),
            "background_neighbors": 1024,
            "independent_acceptance": False,
            "next_gate": "Proposal coverage/ranking cannot qualify release; full paired current0149 and known protection required",
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--all-queries", action="store_true", default=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.all_queries), indent=2))


if __name__ == "__main__":
    main()
