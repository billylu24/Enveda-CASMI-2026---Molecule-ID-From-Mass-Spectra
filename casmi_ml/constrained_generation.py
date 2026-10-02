"""Paired development-only atom-mass ceiling pilot with a frozen decoder."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chemistry import extract_evidence, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.generation_constraints import mass_eos_validator, token_atom_masses
from casmi_ml.generation_experiment import load_model, validate_generated
from casmi_ml.generation_sampling import condition_for_group, sampling_seed
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ENCODER, ROOT, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure


def run(output, limit=200, exact_eos=False):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = ROOT / "generation/smiles_42/model.pt"
    baseline_path = ROOT / "generation/researchdev_samples128_limitall_stable_v2.json"
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "constraint": "conservative_atom_mass_ceiling_v1",
            "exact_eos": exact_eos,
            "checkpoint_sha256": digest(checkpoint),
            "baseline_sha256": digest(baseline_path),
            "encoder_sha256": digest(ENCODER),
            "input_sha256": digest(ROOT / "researchdev.parquet"),
            "samples": 128,
            "limit": limit,
            "temperature": 0.8,
            "sampling": "spectrum_hash_v1_and_shared_group_forward_v2",
            "oracle_formula": False,
            "holdout_used": False,
            "scope": "Paired pilot diagnostics; no structural release eligibility",
        },
    )
    if (output / "report.json").exists():
        return json.loads((output / "report.json").read_text())
    configure(42, threads=4)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    baseline = {r["key"]: r for r in json.loads(baseline_path.read_text())}
    partial = output / "samples.partial.json"
    rows = json.loads(partial.read_text()) if partial.exists() else []
    done = {r["key"] for r in rows}
    with_budget = StageBudget(output, "generation", "mass_ceiling", 86400)
    started = time.monotonic()
    try:
        decoder, formula, vocabulary = load_model(checkpoint, device)
        encoder, saved = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device)
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        groups = list(frame.groupby("inchikey14", sort=True).indices.items())[:limit]
        masses = token_atom_masses(vocabulary)
        for i, (key, indices) in enumerate(groups, 1):
            if key in done:
                continue
            group = frame.iloc[indices]
            # Conditioner and seed receive no truth/formula/key fields.
            query = group.drop(
                columns=[
                    c
                    for c in ["inchikey14", "normalized_smiles", "molecular_formula"]
                    if c in group
                ]
            )
            values = [neutral_mass(r) for r in query.to_dict("records")]
            mass = (
                float(np.median([m for m in values if m is not None]))
                if any(m is not None for m in values)
                else None
            )
            condition = condition_for_group(
                encoder, saved["preprocessing"], query, device
            )
            hypotheses = formula.hypotheses(condition)
            with torch.inference_mode():
                sequences, logp, finished = decoder.generate(
                    torch.cat([condition, formula.soft(condition)], 1),
                    128,
                    generator=torch.Generator(device=device).manual_seed(
                        sampling_seed(query)
                    ),
                    deadline=with_budget.started + with_budget.allowance,
                    token_masses=masses if mass is not None else None,
                    neutral_mass=mass,
                    eos_validator=mass_eos_validator(vocabulary, mass)
                    if exact_eos and mass is not None
                    else None,
                )
            candidates, stats = validate_generated(
                sequences.cpu().tolist(),
                logp.cpu().tolist(),
                finished.cpu().tolist(),
                vocabulary,
                mass,
                hypotheses,
                [extract_evidence(r) for r in query.to_dict("records")],
            )
            rows.append({"key": key, "candidates": candidates, "statistics": stats})
            if i % 20 == 0:
                write_json(partial, rows)
                print(
                    "constrained", i, "seconds", time.monotonic() - started, flush=True
                )
        write_json(output / "samples.json", rows)
        paired = [baseline[r["key"]] for r in rows]

        def summarize(values):
            stats = {
                name: sum(r["statistics"][name] for r in values)
                for name in [
                    "samples",
                    "terminated",
                    "valid",
                    "mass_matching",
                    "unique_mass_matching",
                ]
            }
            stats["queries_with_candidates"] = sum(
                bool(r["candidates"]) for r in values
            )
            stats["exact_structure_hits"] = sum(
                any(c["key"] == r["key"] for c in r["candidates"]) for r in values
            )
            return stats

        result = {
            "molecules": len(rows),
            "baseline": summarize(paired),
            "constrained": summarize(rows),
            "seconds": time.monotonic() - started,
            "independent_acceptance": False,
            "diagnostic_only": True,
            "truth_used_only_in_metrics": True,
        }
        write_json(output / "report.json", result)
        return result
    finally:
        with_budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=200)
    p.add_argument("--exact-eos", action="store_true")
    a = p.parse_args()
    if a.limit < 1:
        p.error("--limit must be positive")
    print(json.dumps(run(a.output, a.limit, a.exact_eos), indent=2))


if __name__ == "__main__":
    main()
