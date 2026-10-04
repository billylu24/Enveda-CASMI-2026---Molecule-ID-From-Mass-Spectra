"""Fixed128 local pilot: preserve baseline trajectories and add mass-aware proposals."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chemistry import composition_mass, extract_evidence, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.generation_experiment import (
    FormulaPredictor,
    count_formula,
    formula_counts,
    load_model,
    validate_generated,
)
from casmi_ml.generation_sampling import condition_for_group, sampling_seed
from casmi_ml.generation_temperature_experiment import CHECKPOINT, summarize
from casmi_ml.mass_formula_proposal_audit import mass_proposals
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

FORMULA = Path(
    "artifacts/research_loop/rounds/0415_mass_formula_training_proposal_audit"
)


def mixed_condition(z, original_soft, proposals, samples=128):
    if samples != 128:
        raise ValueError("Fixed128 matched sample budget required")
    baseline = torch.cat([z, original_soft], 1).expand(samples, -1).clone()
    if not proposals:
        return baseline
    for slot in range(64):
        counts = formula_counts(proposals[slot % len(proposals)])
        if counts is None:
            raise ValueError("Supported nonoracle proposal formula required")
        baseline[64 + slot, z.shape[1] :] = (
            torch.as_tensor(counts, device=z.device, dtype=z.dtype) / 32.0
        )
    return baseline


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU pilot required")
    output = Path(output)
    evidence = json.loads((FORMULA / "report.json").read_text())
    prior = json.loads((FORMULA / "protocol.json").read_text())
    if (
        prior["source_sha256"] != digest("casmi_ml/mass_formula_proposal_audit.py")
        or prior["train_sha256"] != digest(TRAIN)
        or evidence["epochs"] != 12
        or evidence["heldout_molecules"] != 12006
    ):
        raise ValueError("Source-bound fixed training formula head required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "formula_head_sha256": digest(FORMULA / "private_formula.pt"),
            "formula_protocol_sha256": digest(FORMULA / "protocol.json"),
            "formula_report_sha256": digest(FORMULA / "report.json"),
            "decoder_checkpoint_sha256": digest(CHECKPOINT),
            "baseline_generated_sha256": digest(GENERATED),
            "training_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "encoder_sha256": digest(ENCODER),
            "sources": {
                n: digest("casmi_ml/" + n + ".py")
                for n in (
                    "generation_experiment",
                    "generation_sampling",
                    "research_models",
                    "mass_formula_proposal_audit",
                    "forward_diagnostic_training_audit",
                )
            },
            "rule": "Fixed first200 original sorted development keys, original128samples/.8temperature/per-spectrum seed/decoder/encoder/validation. First64 rows retain original softformula condition exactly; remaining64 use evenly allocated observed-mass compatible top5 fit-only formula count proposals /32. Compare original128 baseline, neural-mass proposals and matched mass-only training-frequency proposals. All arms128 samples, no extra sampling/token constraints/label proposal inputs; missing proposals all128 original. Original validation hypotheses retained for candidate metadata, no ranking model changes. Require regenerated baseline stats/candidates/formula hypotheses equal original cache on all200 before summary acceptance. Pilot is repeated development; no fresh independent acceptance/activation. Fixed all arms before development metrics, do not tune blend/temperature/sample count on results.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    configure(42, threads=4)
    budget = StageBudget(output, "generation", str(output), 1800)
    start = time.monotonic()
    try:
        device = torch.device("cuda")
        decoder, original_formula, vocabulary = load_model(CHECKPOINT, device)
        saved = torch.load(
            FORMULA / "private_formula.pt", map_location="cpu", weights_only=True
        )
        head = FormulaPredictor(saved["condition_dim"]).to(device)
        head.load_state_dict(saved["formula"])
        head.eval()
        encoder, encoder_checkpoint = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device)
        train = pd.read_parquet(TRAIN, columns=["inchikey14", "molecular_formula"])
        first = train.groupby("inchikey14", sort=True).first()
        fit_keys = ~heldout_mask(list(first.index))
        parsed = [formula_counts(f) for f in first.molecular_formula]
        fit_formulas = [
            count_formula(v)
            for v, allowed in zip(parsed, fit_keys)
            if allowed and v is not None
        ]
        from collections import Counter

        counts = Counter(fit_formulas)
        formulas = sorted(counts, key=lambda f: (composition_mass(f), f))
        masses = np.asarray([composition_mass(f) for f in formulas])
        counts_matrix = np.stack([formula_counts(f) for f in formulas])
        frequency = np.log(
            np.asarray([counts[f] / len(fit_formulas) for f in formulas])
        )
        if len(formulas) != evidence["empirical_fit_formulas"]:
            raise ValueError("Fit-only empirical formula catalog differs")
        expected = {r["key"]: r for r in json.loads(GENERATED.read_text())}
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        groups = list(frame.groupby("inchikey14", sort=True).indices.items())[:200]
        arms = {name: [] for name in ("original", "neural_mass", "frequency_mass")}
        proposal_counts = []
        for index, (key, ids) in enumerate(groups):
            if c.stopped() or not budget.checkpoint():
                raise RuntimeError("Pilot stopped before complete200 matched arms")
            group = frame.iloc[ids].drop(
                columns=["inchikey14", "molecular_formula", "normalized_smiles"],
                errors="ignore",
            )
            z = condition_for_group(
                encoder, encoder_checkpoint["preprocessing"], group, device
            )
            observed = [
                m
                for r in group.to_dict("records")
                if (m := neutral_mass(r)) is not None
            ]
            mass = float(np.median(observed)) if observed else None
            evidences = [extract_evidence(r) for r in group.to_dict("records")]
            with torch.inference_mode():
                logp = head(z).log_softmax(-1)[0].cpu().numpy()
                proposals, n = mass_proposals(
                    logp, mass, masses, counts_matrix, formulas
                )
                control, other_n = mass_proposals(
                    logp,
                    mass,
                    masses,
                    counts_matrix,
                    formulas,
                    formula_log_prior=frequency,
                )
                if n != other_n:
                    raise ValueError("Matched proposal support differs")
                proposal_counts.append(n)
                hypotheses = original_formula.hypotheses(z, k=5)
                soft = original_formula.soft(z)
                for name, choices in (
                    ("original", []),
                    ("neural_mass", proposals),
                    ("frequency_mass", control),
                ):
                    condition = mixed_condition(z, soft, choices)
                    generator = torch.Generator(device=device).manual_seed(
                        sampling_seed(group)
                    )
                    sequences, likelihood, finished = decoder.generate(
                        condition,
                        128,
                        temperature=0.8,
                        generator=generator,
                        deadline=budget.started + budget.allowance,
                    )
                    candidates, stats = validate_generated(
                        sequences.cpu().tolist(),
                        likelihood.cpu().tolist(),
                        finished.cpu().tolist(),
                        vocabulary,
                        mass,
                        hypotheses,
                        evidences,
                        track_frequency=True,
                    )
                    row = {
                        "key": key,
                        "candidates": candidates,
                        "statistics": stats,
                        "formula_hypotheses": hypotheses,
                        "fragment_status": [],
                        "oracle_formula": False,
                    }
                    arms[name].append(row)
            if any(
                arms["original"][-1][field] != expected[key][field]
                for field in ("candidates", "statistics", "formula_hypotheses")
            ):
                write_json(output / "private_partial_arms.json", arms)
                raise ValueError("Fresh pilot baseline differs from original cache")
            if (index + 1) % 10 == 0:
                write_json(output / "private_partial_arms.json", arms)
                print("Completed matched pilot", index + 1, flush=True)
        write_json(output / "private_complete_arms.json", arms)
        report = {
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
            "molecules": 200,
            "all200_original_candidates_stats_hypotheses_exact": True,
            "samples_per_query_per_arm": 128,
            "original_soft_condition_rows_per_query": 64,
            "queries_with_proposals": int(np.count_nonzero(proposal_counts)),
            "arms": {name: summarize(rows) for name, rows in arms.items()},
            "seconds": time.monotonic() - start,
            "scope": "Repeated-development fixed paired structural recall pilot. Formula targets never enter proposal inputs. No structure-ranking/independent accuracy acceptance, physics or activation; best236 retained.",
        }
        write_json(output / "report.json", report)
        print(json.dumps(report), flush=True)
        return report
    finally:
        budget.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
