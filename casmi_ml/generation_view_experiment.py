"""Fixed-budget mean/single-spectrum conditioning pilot using the frozen decoder."""

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
from casmi_ml.generation_views import generation_views
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def generate_for_group(
    decoder,
    formula,
    vocabulary,
    encoder,
    preprocessing,
    group,
    device,
    deadline,
    control=False,
):
    sequence_parts, probability_parts, finished_parts, hypotheses = [], [], [], []
    views = generation_views(group)
    if control:
        views = [(group, count) for _, count in views]
    for index, (view, count) in enumerate(views):
        condition = condition_for_group(encoder, preprocessing, view, device)
        # The full query identifies the mixture; view tags separate RNG streams.
        seed = int.from_bytes(
            hashlib.sha256(
                f"views-v1:{sampling_seed(group)}:{index}".encode()
            ).digest()[:8],
            "big",
        ) % (2**63 - 1)
        with torch.inference_mode():
            sequence, logp, finished = decoder.generate(
                torch.cat([condition, formula.soft(condition)], 1),
                count,
                generator=torch.Generator(device=device).manual_seed(seed),
                deadline=deadline,
            )
        sequence_parts.extend(sequence.cpu().tolist())
        probability_parts.extend(logp.cpu().tolist())
        finished_parts.extend(finished.cpu().tolist())
        hypotheses.extend(formula.hypotheses(condition))
    masses = [neutral_mass(row) for row in group.to_dict("records")]
    mass = (
        float(np.median([m for m in masses if m is not None]))
        if any(m is not None for m in masses)
        else None
    )
    return validate_generated(
        sequence_parts,
        probability_parts,
        finished_parts,
        vocabulary,
        mass,
        hypotheses,
        [extract_evidence(row) for row in group.to_dict("records")],
    )


def run(output, limit=200):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "generation/smiles_42/model.pt"
    baseline_path = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "view_selection_sha256": digest(
                Path(__file__).with_name("generation_views.py")
            ),
            "sampler": "views_mean64_effective_peak_single64_v1",
            "checkpoint_sha256": digest(checkpoint),
            "encoder_sha256": digest(ENCODER),
            "baseline_sha256": digest(baseline_path),
            "input_sha256": digest(ROOT / "researchdev.parquet"),
            "samples": 128,
            "arms": ["split_mean64_mean64_control", "mean64_single64"],
            "limit": limit,
            "holdout_used": False,
            "oracle_formula": False,
        },
    )
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    configure(42, threads=4)
    budget = StageBudget(output, "generation", "views", 86400)
    started = time.monotonic()
    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        decoder, formula, vocabulary = load_model(checkpoint, device)
        encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device)
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        partial = output / "samples.partial.json"
        rows = json.loads(partial.read_text()) if partial.exists() else []
        done = {row["key"] for row in rows}
        for i, (key, ids) in enumerate(
            list(frame.groupby("inchikey14", sort=True).indices.items())[:limit], 1
        ):
            if key in done:
                continue
            group = frame.iloc[ids].drop(
                columns=[
                    c
                    for c in ["inchikey14", "normalized_smiles", "molecular_formula"]
                    if c in frame
                ]
            )
            candidates, stats = generate_for_group(
                decoder,
                formula,
                vocabulary,
                encoder,
                saved["preprocessing"],
                group,
                device,
                budget.started + budget.allowance,
            )
            control_candidates, control_stats = generate_for_group(
                decoder,
                formula,
                vocabulary,
                encoder,
                saved["preprocessing"],
                group,
                device,
                budget.started + budget.allowance,
                control=True,
            )
            rows.append(
                {
                    "key": key,
                    "candidates": candidates,
                    "statistics": stats,
                    "control_candidates": control_candidates,
                    "control_statistics": control_stats,
                }
            )
            if i % 20 == 0:
                write_json(partial, rows)
                print("views", i, "seconds", time.monotonic() - started, flush=True)
        write_json(output / "samples.json", rows)
        baseline = {r["key"]: r for r in json.loads(baseline_path.read_text())}

        def summarize(values):
            result = {
                name: sum(r["statistics"][name] for r in values)
                for name in [
                    "samples",
                    "terminated",
                    "valid",
                    "mass_matching",
                    "unique_mass_matching",
                ]
            }
            result.update(
                queries_with_candidates=sum(bool(r["candidates"]) for r in values),
                exact_structure_hits=sum(
                    any(c["key"] == r["key"] for c in r["candidates"]) for r in values
                ),
            )
            return result

        result = {
            "molecules": len(rows),
            "baseline": summarize([baseline[r["key"]] for r in rows]),
            "split_control": summarize(
                [
                    {
                        "key": row["key"],
                        "candidates": row["control_candidates"],
                        "statistics": row["control_statistics"],
                    }
                    for row in rows
                ]
            ),
            "view_mixture": summarize(rows),
            "seconds": time.monotonic() - started,
            "diagnostic_only": True,
            "independent_acceptance": False,
            "truth_used_only_in_metrics": True,
            "sampling_budget": "128 trajectories total per query in both arms; different frozen RNG streams",
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=200)
    a = p.parse_args()
    if a.limit < 1:
        p.error("--limit must be positive")
    print(json.dumps(run(a.output, a.limit), indent=2))


if __name__ == "__main__":
    main()
