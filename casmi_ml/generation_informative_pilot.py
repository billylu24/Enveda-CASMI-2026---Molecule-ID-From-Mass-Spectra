"""Fixed-budget contrastive decoding against a training-only latent spectrum prior."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chemistry import extract_evidence, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.generation_experiment import load_model, validate_generated
from casmi_ml.generation_sampling import condition_for_group, sampling_seed
from casmi_ml.generation_temperature_experiment import BASELINE, CHECKPOINT, summarize
from casmi_ml.generation_views import informative_spectrum
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def run(output, limit=200):
    if limit < 1:
        raise ValueError("Positive pilot limit required")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "decoder_source_sha256": digest("casmi_ml/research_models.py"),
            "generator_sha256": digest(CHECKPOINT),
            "encoder_sha256": digest(ENCODER),
            "baseline_sha256": digest(BASELINE),
            "limit": limit,
            "samples": 128,
            "temperature": 0.8,
            "condition_view": "informative_spectrum_v1; full-group seed and median mass preserved",
            "view_source_sha256": digest(
                Path(informative_spectrum.__code__.co_filename)
            ),
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
            "cohort": "repeated_development",
            "new_training": False,
        },
    )
    configure(42, threads=4)
    budget = StageBudget(output, "generation", "informative_sampling", 86400)
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder, formula, vocabulary = load_model(CHECKPOINT, device)
        encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device).eval()
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        groups = list(frame.groupby("inchikey14", sort=True).indices.items())[:limit]
        generated = []
        for key, ids in groups:
            group = frame.iloc[ids]
            condition = condition_for_group(
                encoder, saved["preprocessing"], informative_spectrum(group), device
            )
            with torch.inference_mode():
                counts = formula.soft(condition)
                augmented = torch.cat([condition, counts], 1)
                seq, logp, finished = decoder.generate(
                    augmented,
                    128,
                    temperature=0.8,
                    generator=torch.Generator(device=device).manual_seed(
                        sampling_seed(group)
                    ),
                    deadline=budget.started + budget.allowance,
                )
                hypotheses = formula.hypotheses(condition)
            masses = [
                m
                for row in group.to_dict("records")
                if (m := neutral_mass(row)) is not None
            ]
            evidence = [extract_evidence(row) for row in group.to_dict("records")]
            candidates, stats = validate_generated(
                seq.cpu().tolist(),
                logp.cpu().tolist(),
                finished.cpu().tolist(),
                vocabulary,
                float(np.median(masses)) if masses else None,
                hypotheses,
                evidence,
                track_frequency=True,
            )
            generated.append(
                {"key": key, "candidates": candidates, "statistics": stats}
            )
            if len(generated) % 25 == 0:
                write_json(output / "samples.partial.json", generated)
                print(
                    "informative_sampled", len(generated), "/", len(groups), flush=True
                )
        write_json(output / "samples.json", generated)
    finally:
        budget.close()
    original = {r["key"]: r for r in json.loads(BASELINE.read_text())}
    result = {
        "diagnostic_only": True,
        "molecules": len(generated),
        "baseline": summarize([original[r["key"]] for r in generated]),
        "selected": summarize(generated),
        "condition_view": "informative_spectrum_v1",
        "independent_acceptance": False,
        "scope": "Fixed200 informative-spectrum condition; no submission eligibility",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()
