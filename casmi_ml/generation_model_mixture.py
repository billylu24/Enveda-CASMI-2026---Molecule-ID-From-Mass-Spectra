"""Paired 64+64 decoder mixture pilot, sharing the first half across arms."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chemistry import extract_evidence, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.generation_experiment import load_model, validate_generated
from casmi_ml.generation_sampling import condition_for_group, sampling_seed
from casmi_ml.generation_temperature_experiment import BASELINE, CHECKPOINT, summarize
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def stream_seed(group, index):
    if index not in [0, 1]:
        raise ValueError("A 64+64 mixture has exactly two streams")
    return int.from_bytes(
        hashlib.sha256(
            f"decoder-mixture-v1:{sampling_seed(group)}:{index}".encode()
        ).digest()[:8],
        "big",
    ) % (2**63 - 1)


def combine(parts, vocabulary, mass, hypotheses, evidences):
    values = [[], [], []]
    for part in parts:
        for accumulated, tensor in zip(values, part):
            accumulated.extend(tensor.cpu().tolist())
    return validate_generated(*values, vocabulary, mass, hypotheses, evidences)


def run(output, limit=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if limit < 1:
        raise ValueError("Positive pilot size required")
    original_path = ROOT / "generation/smiles_42/model.pt"
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "selected_checkpoint_sha256": digest(CHECKPOINT),
            "original_checkpoint_sha256": digest(original_path),
            "baseline_sha256": digest(BASELINE),
            "encoder_sha256": digest(ENCODER),
            "input_sha256": digest(ROOT / "researchdev.parquet"),
            "limit": limit,
            "samples_per_arm": 128,
            "samples_per_part": 64,
            "temperature": 0.8,
            "arms": ["selected64_selected64_control", "selected64_original64_mixture"],
            "first_half_shared": True,
            "second_half_rng_shared_between_models": True,
            "sampler": "decoder-mixture-v1_spectrum_hash_stream_index",
            "conditions_shared": True,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
        },
    )
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    configure(42, threads=4)
    budget = StageBudget(output, "generation", "decoder_mixture_sampling", 86400)
    start = time.monotonic()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        selected, formula, vocabulary = load_model(CHECKPOINT, device)
        original, original_formula, original_vocabulary = load_model(
            original_path, device
        )
        if vocabulary.tokens != original_vocabulary.tokens or any(
            not torch.equal(value, original_formula.state_dict()[name])
            for name, value in formula.state_dict().items()
        ):
            raise ValueError(
                "Mixture requires identical formula predictor and vocabulary"
            )
        encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device)
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        partial = output / "samples.partial.json"
        rows = json.loads(partial.read_text()) if partial.exists() else []
        done = {r["key"] for r in rows}
        for i, (key, ids) in enumerate(
            list(frame.groupby("inchikey14", sort=True).indices.items())[:limit], 1
        ):
            if key in done:
                continue
            group = frame.iloc[ids].drop(
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
                encoder, saved["preprocessing"], group, device
            )
            with torch.inference_mode():
                z = torch.cat([condition, formula.soft(condition)], 1)
                hypotheses = formula.hypotheses(condition)
                parts = [
                    decoder.generate(
                        z,
                        64,
                        generator=torch.Generator(device=device).manual_seed(
                            stream_seed(group, index)
                        ),
                        deadline=budget.started + budget.allowance,
                    )
                    for decoder, index in [(selected, 0), (selected, 1), (original, 1)]
                ]
            masses = [neutral_mass(r) for r in group.to_dict("records")]
            mass = (
                float(np.median([m for m in masses if m is not None]))
                if any(m is not None for m in masses)
                else None
            )
            evidences = [extract_evidence(r) for r in group.to_dict("records")]
            control, control_stats = combine(
                parts[:2], vocabulary, mass, hypotheses, evidences
            )
            mixture, mixture_stats = combine(
                [parts[0], parts[2]], vocabulary, mass, hypotheses, evidences
            )
            rows.append(
                {
                    "key": key,
                    "candidates": mixture,
                    "statistics": mixture_stats,
                    "control_candidates": control,
                    "control_statistics": control_stats,
                }
            )
            if i % 10 == 0:
                write_json(partial, rows)
                print("mixtures", i, "seconds", time.monotonic() - start, flush=True)
        write_json(output / "samples.json", rows)
        baseline = {r["key"]: r for r in json.loads(BASELINE.read_text())}
        result = {
            "molecules": len(rows),
            "baseline": summarize([baseline[r["key"]] for r in rows]),
            "split_control": summarize(
                [
                    {
                        "key": r["key"],
                        "candidates": r["control_candidates"],
                        "statistics": r["control_statistics"],
                    }
                    for r in rows
                ]
            ),
            "model_mixture": summarize(rows),
            "seconds": time.monotonic() - start,
            "diagnostic_only": True,
            "independent_acceptance": False,
            "sampling_budget": "128 trajectories per arm; shared first64; paired second64 random stream; 192 total computational trajectories per query",
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.limit), indent=2))


if __name__ == "__main__":
    main()
