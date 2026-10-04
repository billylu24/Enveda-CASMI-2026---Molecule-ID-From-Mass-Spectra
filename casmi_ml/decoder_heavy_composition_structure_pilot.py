"""Fixed128 matched structural pilot for the frozen heavy-composition objective."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.chemistry import extract_evidence, neutral_mass
from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.generation_experiment import (
    load_model,
    validate_generated,
)
from casmi_ml.generation_sampling import condition_for_group, sampling_seed
from casmi_ml.generation_temperature_experiment import CHECKPOINT, summarize
from casmi_ml.mass_formula_generation_pilot import mixed_condition
from casmi_ml.metfrag import digest
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, ROOT, TRAIN, freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

TRAINED = Path(
    "artifacts/research_loop/rounds/0437_decoder_heavy_composition_deterministic_training_audit"
)


def run(output):
    c = Controller()
    if not c.local_only() or not torch.cuda.is_available():
        raise ValueError("Local-only GPU structural pilot required")
    output = Path(output)
    trained = TRAINED
    training_report = json.loads((trained / "report.json").read_text())
    training_protocol = json.loads((trained / "protocol.json").read_text())
    if (
        training_protocol["source_sha256"]
        != digest("casmi_ml/decoder_heavy_composition_training_audit.py")
        or not training_report[
            "all_deterministic_control_replica_tensors_and_assessments_exact"
        ]
        or training_report["heldout_molecules"] != 12006
    ):
        raise ValueError(
            "Source-bound complete deterministic replicatedcontrol training required"
        )
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_report_sha256": digest(trained / "report.json"),
            "training_protocol_sha256": digest(trained / "protocol.json"),
            "decoder_arms": {
                name: digest(trained / ("private_" + name + "_decoder.pt"))
                for name in ("soft_control", "heavy_auxiliary")
            },
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
                    "decoder_heavy_composition_training_audit",
                    "forward_diagnostic_training_audit",
                )
            },
            "rule": "Prespecified same sorted first200 repeateddevelopment, three arms original/control/heavyaux all128/.8temperature/sameperqueryseed/originalencoder/formulahead/hypotheses/meanobservablecondition/originalvalidation. Training437 deterministic control replicas allweight/assessment exact; treatmentonly0.1 normalizedheavySmoothL1/beta.01 fixedfittrainingtargets, no query trueformulas/proposals/constraints/increasedsamples/retraining/checkpoint selection. Strip labels before conditioning. Original200 fullcandidates/stats/hypotheses must exactly reproduce originalcache before aggregate. Measure exactgraph overlap and valid/massunique/candidatequery, not fullranking acceptance. No tuning after pilot, keep236 if no gain.",
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
        decoders = {"original": decoder}
        for name in ("soft_control", "heavy_auxiliary"):
            model, _, _ = load_model(CHECKPOINT, device)
            saved_decoder = torch.load(
                trained / ("private_" + name + "_decoder.pt"),
                map_location="cpu",
                weights_only=True,
            )
            if saved_decoder["initial_checkpoint_sha256"] != digest(CHECKPOINT):
                raise ValueError("Matched training initial decoder differs")
            model.load_state_dict(saved_decoder["decoder"])
            decoders[name] = model.eval()
        encoder, encoder_checkpoint = load_deployment_checkpoint(ENCODER, "scale")
        encoder.to(device)
        expected = {r["key"]: r for r in json.loads(GENERATED.read_text())}
        frame = pd.read_parquet(ROOT / "researchdev.parquet")
        groups = list(frame.groupby("inchikey14", sort=True).indices.items())[:200]
        arms = {
            name: []
            for name in ("original", "soft_control_soft", "heavy_auxiliary_soft")
        }
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
                hypotheses = original_formula.hypotheses(z, k=5)
                soft = original_formula.soft(z)
                for name, decoder_name in [
                    ("original", "original"),
                    ("soft_control_soft", "soft_control"),
                    ("heavy_auxiliary_soft", "heavy_auxiliary"),
                ]:
                    condition = mixed_condition(z, soft, [])
                    generator = torch.Generator(device=device).manual_seed(
                        sampling_seed(group)
                    )
                    sequences, likelihood, finished = decoders[decoder_name].generate(
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
            "original_soft_condition_rows_per_query": 128,
            "arms": {name: summarize(rows) for name, rows in arms.items()},
            "seconds": time.monotonic() - start,
            "scope": "Fixed three-arm repeateddevelopment structural pilot after frozen437 deterministic replicated control training. Query formula targets never enter conditions. Compare training composition auxiliary objective against matched soft-only finetuning, no structure-ranking/independent accuracy acceptance, physics or activation; best236 retained.",
        }
        hits = {
            name: {
                r["key"]
                for r in rows
                if any(c["key"] == r["key"] for c in r["candidates"])
            }
            for name, rows in arms.items()
        }
        report["graph_hit_overlap"] = {
            name: {
                "preserved_original": len(hits[name] & hits["original"]),
                "new_vs_original": len(hits[name] - hits["original"]),
                "lost_original": len(hits["original"] - hits[name]),
            }
            for name in arms
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
