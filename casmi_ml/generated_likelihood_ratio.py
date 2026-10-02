"""Diagnostic canonical sequence likelihood ratios on frozen generated structures."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chemistry import rerank
from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import GENERATED, SCORES, combined_order
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

VARIANTS = {
    "baseline": None,
    "canonical": 0.0,
    "ratio025": 0.25,
    "ratio05": 0.5,
    "ratio1": 1.0,
}


def log_likelihood(logits, targets, temperature=0.8):
    logits = logits.float() / temperature
    logits[:, :, [0, 1, 3]] = float("-inf")
    valid = targets != 0
    # Padding is assigned a legal gather index then removed from the sum.
    values = (
        logits.log_softmax(-1)
        .gather(-1, targets.masked_fill(~valid, 2).unsqueeze(-1))
        .squeeze(-1)
    )
    return values.masked_fill(~valid, 0).sum(1) / valid.sum(1).clamp_min(1)


@torch.inference_mode()
def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "generator_sha256": digest(CHECKPOINT),
            "encoder_sha256": digest(ENCODER),
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "training_sha256": digest(TRAIN),
            "variants": VARIANTS,
            "temperature": 0.8,
            "sequence": "canonical generated SMILES; teacher forced mean log probability includingEOS; special-token mask matches sampling",
            "prior": "equally molecule-weighted60000 training encoder mean; query metadata,mass,predicted formula preserved",
            "ordering": "rerank frozen token-calibrated+critic ordering with weight0.5 and query_logp -ratio_weight*prior_logp",
            "fallback": "unchanged baseline if canonical vocabulary or length rejects any candidate",
            "new_sampling": False,
            "new_training": False,
            "holdout_used": False,
            "cohort": "repeated_development",
            "truth_used_only_in_metrics": True,
            "scope": "generated-only2000 diagnostics; no release gate or combined retrieval claim",
        },
    )
    configure(42, threads=4)
    budget = StageBudget(output, "generation", "likelihood_ratio", 3600)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder, formula, vocabulary = load_model(CHECKPOINT, device)
        encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device).eval()
        latent = torch.from_numpy(training_prior()).to(device)
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        training = set(pd.read_parquet(TRAIN, columns=["inchikey14"]).inchikey14)
        if training & set(frame.inchikey14):
            raise ValueError("Training/development overlap")
        groups = frame.groupby("inchikey14", sort=True).indices
        generated = json.loads(GENERATED.read_text())
        reference = json.loads(SCORES.read_text())
        if len(generated) != 2000 or {r["key"] for r in generated} != set(groups):
            raise ValueError("Full2000 keys required")
        ranks = {name: {} for name in VARIANTS}
        pools, result_scores, fallbacks = {}, {}, 0
        for i, record in enumerate(generated, 1):
            if not budget.checkpoint():
                raise TimeoutError("Likelihood budget exhausted")
            key, candidates = record["key"], record["candidates"]
            base = combined_order(
                candidates, reference.get(key, {}).get("fingerprint", {}), (1, 0, 0.5)
            )
            pools[key] = base
            for name in VARIANTS:
                ranks[name][key] = base
            if len(candidates) < 2:
                continue
            sequences = [vocabulary.encode(c["smiles"]) for c in candidates]
            if any(v is None for v in sequences):
                fallbacks += 1
                continue
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
            augmented = torch.cat([condition, formula.soft(condition)], 1)
            prior = prior_input(augmented, latent)
            scores = {}
            for start in range(0, len(candidates), 16):
                batch = sequences[start : start + 16]
                tokens = torch.nn.utils.rnn.pad_sequence(
                    [torch.tensor(v, device=device) for v in batch],
                    batch_first=True,
                    padding_value=0,
                )
                actual = log_likelihood(
                    decoder(tokens[:, :-1], augmented.expand(len(batch), -1)),
                    tokens[:, 1:],
                )
                background = log_likelihood(
                    decoder(tokens[:, :-1], prior.expand(len(batch), -1)), tokens[:, 1:]
                )
                for c, q, p in zip(
                    candidates[start : start + 16],
                    actual.cpu().tolist(),
                    background.cpu().tolist(),
                ):
                    if not np.isfinite([q, p]).all():
                        raise ValueError("Nonfinite sequence scores")
                    scores[c["key"]] = {"query": q, "prior": p}
            result_scores[key] = scores
            for name, weight in VARIANTS.items():
                if weight is not None:
                    ranks[name][key] = rerank(
                        base,
                        {},
                        [],
                        0.5,
                        top_n=len(base),
                        fragment_scores={
                            k: v["query"] - weight * v["prior"]
                            for k, v in scores.items()
                        },
                    )
            if i % 100 == 0:
                print("likelihood_scored", i, "/2000", flush=True)
        write_json(output / "scores.json", result_scores)
        report = {}
        for name in VARIANTS:
            result, per = metrics(ranks[name], pools)
            report[name] = result
            per.to_csv(output / f"generated_{name}.csv", index=False)
        summary = {
            "diagnostic_only": True,
            "molecules": 2000,
            "generated_only": report,
            "queries_scored": len(result_scores),
            "canonical_fallbacks": fallbacks,
            "independent_acceptance": False,
            "scope": "Fixed generated pool only; combined retrieval and known protection not evaluated",
        }
        write_json(output / "report.json", summary)
        return summary
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))


if __name__ == "__main__":
    main()
