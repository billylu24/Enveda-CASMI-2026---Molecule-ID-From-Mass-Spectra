"""Frozen decoder as a conditional scorer of mass-filtered external structures."""

import argparse
import json
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.generated_likelihood_ratio import log_likelihood
from casmi_ml.generation_contrastive_pilot import prior_input, training_prior
from casmi_ml.generation_experiment import load_model
from casmi_ml.generation_sampling import condition_for_group
from casmi_ml.generation_temperature_experiment import CHECKPOINT
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

VARIANTS = {"native": None, "query": 0.0, "conditional_ratio": 1.0}
PROPOSALS = Path(
    "artifacts/research_loop/rounds/0071_chembl_candidate_slots/proposals.json"
)


def reorder_scoreable(base, scores, ratio):
    valid = [k for k in base if k in scores]
    order = iter(
        rerank(
            valid,
            {},
            [],
            0.5,
            top_n=max(1, len(valid)),
            fragment_scores={
                k: scores[k]["query"] - ratio * scores[k]["prior"] for k in valid
            },
        )
    )
    # Unsupported tokens/length leave their original slots unchanged.
    return [next(order) if k in scores else k for k in base]


@torch.inference_mode()
def run(output, limit=200):
    if limit not in (200, 2000):
        raise ValueError("Diagnostic limit must be 200 or 2000")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "generator_sha256": digest(CHECKPOINT),
            "encoder_sha256": digest(ENCODER),
            "catalog_sha256": digest(DERIVED),
            "proposals_sha256": digest(PROPOSALS),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "training_sha256": digest(TRAIN),
            "limit": limit,
            "candidate_limit": 500,
            "variants": VARIANTS,
            "temperature": 0.8,
            "fusion_weight": 0.5,
            "sequence": "Canonical external SMILES teacher forced mean log probability, including EOS, same special-token mask as sampling",
            "prior": "Equally weighted 60K training-only mean encoder latent; actual query metadata, mass and soft predicted formula retained",
            "unsupported": "Preserve exact original candidate slots and only permute scoreable candidates",
            "new_training": False,
            "new_sampling": False,
            "truth_used_only_in_metrics": True,
            "cohort": "repeated_development",
            "holdout_used": False,
            "stage_seconds": 3600,
            "scope": "External candidate ranking diagnostic only; no combined ranking or release eligibility",
        },
    )
    configure(42, threads=4)
    budget = StageBudget(output, "generation", "external_conditional_scoring", 3600)
    started = time.monotonic()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder, formula, vocabulary = load_model(CHECKPOINT, device)
        encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device).eval()
        prior_latent = torch.from_numpy(training_prior()).to(device)
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        groups = frame.groupby("inchikey14", sort=True).indices
        training_keys = set(pd.read_parquet(TRAIN, columns=["inchikey14"]).inchikey14)
        if training_keys & set(groups):
            raise ValueError("Training/development overlap")
        proposals = json.loads(PROPOSALS.read_text())
        keys = sorted(groups)[:limit]
        wanted = {k for key in keys for k in proposals.get(key, [])[:500]}
        catalog = pd.read_parquet(DERIVED, columns=["inchikey14", "normalized_smiles"])
        lookup = (
            catalog.loc[catalog.inchikey14.isin(wanted)]
            .set_index("inchikey14")
            .normalized_smiles.to_dict()
        )
        del catalog
        cache = output / "scores.json"
        result_scores = json.loads(cache.read_text()) if cache.exists() else {}
        rankings = {v: {} for v in VARIANTS}
        pools = {}
        statistics = {
            "queries": 0,
            "candidates": 0,
            "unscoreable_candidates": 0,
            "scored_candidates": 0,
        }
        for i, key in enumerate(keys, 1):
            if not budget.checkpoint():
                raise TimeoutError("Frozen external sequence-scoring budget exhausted")
            base = proposals.get(key, [])[:500]
            pools[key] = base
            for name in VARIANTS:
                rankings[name][key] = base
            if not base:
                continue
            statistics["queries"] += 1
            statistics["candidates"] += len(base)
            # Complete-query scores are resumable and bound by the frozen protocol.
            scores = result_scores.get(key)
            if scores is None:
                query = frame.iloc[groups[key]].drop(
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
                condition = condition_for_group(
                    encoder, saved["preprocessing"], query, device
                )
                condition = torch.cat([condition, formula.soft(condition)], 1)
                background = prior_input(condition, prior_latent)
                encoded = [(k, vocabulary.encode(lookup[k])) for k in base]
                valid = [
                    (k, sequence) for k, sequence in encoded if sequence is not None
                ]
                scores = {}
                for start in range(0, len(valid), 16):
                    if not budget.checkpoint():
                        raise TimeoutError(
                            "Frozen external sequence-scoring budget exhausted"
                        )
                    batch = valid[start : start + 16]
                    tokens = torch.nn.utils.rnn.pad_sequence(
                        [
                            torch.tensor(sequence, device=device)
                            for _, sequence in batch
                        ],
                        batch_first=True,
                        padding_value=0,
                    )
                    actual = log_likelihood(
                        decoder(tokens[:, :-1], condition.expand(len(batch), -1)),
                        tokens[:, 1:],
                    )
                    prior = log_likelihood(
                        decoder(tokens[:, :-1], background.expand(len(batch), -1)),
                        tokens[:, 1:],
                    )
                    for (candidate, _), q, p in zip(
                        batch, actual.cpu().tolist(), prior.cpu().tolist()
                    ):
                        if not np.isfinite([q, p]).all():
                            raise ValueError("Nonfinite conditional sequence score")
                        scores[candidate] = {"query": q, "prior": p}
                result_scores[key] = scores
                write_json(cache, result_scores)
            statistics["scored_candidates"] += len(scores)
            statistics["unscoreable_candidates"] += len(base) - len(scores)
            for name, ratio in VARIANTS.items():
                if ratio is not None:
                    rankings[name][key] = reorder_scoreable(base, scores, ratio)
            if i % 10 == 0:
                print(
                    "external_sequence_queries",
                    i,
                    "seconds",
                    time.monotonic() - started,
                    flush=True,
                )
        report = {}
        for name in VARIANTS:
            result, per = metrics(rankings[name], pools)
            per.to_csv(output / f"external_{name}.csv", index=False)
            report[name] = result
        summary = {
            "diagnostic_only": True,
            "external_only": report,
            "statistics": statistics,
            "seconds": time.monotonic() - started,
            "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            / 1024,
            "gpu_peak_mib": torch.cuda.max_memory_allocated() / 2**20
            if device.type == "cuda"
            else None,
            "independent_acceptance": False,
            "new_training": False,
            "new_sampling": False,
        }
        write_json(output / "report.json", summary)
        return summary
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, choices=(200, 2000), default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()
