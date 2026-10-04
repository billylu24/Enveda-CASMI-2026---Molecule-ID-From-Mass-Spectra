"""Fixed positive-only same-query conditional candidate contrasts, no new fitting."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.candidate_chemistry_specificity_audit import CACHE, contrast_summary
from casmi_ml.data import category, write_json
from casmi_ml.dino_energy_pairs import energy_signature
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.energy_conditioned_diagnostic_audit import feature_block
from casmi_ml.forward_diagnostic_training_audit import heldout_mask
from casmi_ml.full_gap_candidate_compatibility import evidence_features
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives

AUDIT = Path("artifacts/research_loop/rounds/0383_energy_conditioned_diagnostic_audit")


def candidate_prediction(fps, energies, contexts, model, interaction):
    x = feature_block(fps, energies, contexts, interaction)
    return np.clip((x - model["mean_x"]) @ model["weights"] + model["mean_y"], 0, 1)


def run(output):
    started = time.monotonic()
    c = Controller()
    if not c.local_only():
        raise ValueError("Conditional candidate diagnostic is local-only")
    output = Path(output)
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    report = json.loads((AUDIT / "report.json").read_text())
    if (
        protocol["source_sha256"]
        != digest("casmi_ml/energy_conditioned_diagnostic_audit.py")
        or report["heldout_target_used_for_fit"]
        or report["train_development_overlap"]
    ):
        raise ValueError("Complete source-bound molecule-heldout fixed model required")
    models = {}
    for name in ("additive_ce_context", "ce_interaction"):
        with np.load(AUDIT / ("private_" + name + ".npz")) as saved:
            models[name] = {k: saved[k] for k in ("weights", "mean_x", "mean_y")}
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "hard_table_sha256": digest(HARD),
            "prior_source_sha256": protocol["source_sha256"],
            "prior_report_sha256": digest(AUDIT / "report.json"),
            "vocabulary_sha256": digest(AUDIT / "private_context_vocabulary.json"),
            "models_sha256": {
                name: digest(AUDIT / ("private_" + name + ".npz")) for name in models
            },
            "rule": "Reuse both383 fitted models, no retraining. Original molecule-heldout queries singletonCE, bothheldout mass-compatible original pinned15 candidates. For each positive and negative candidate, condition prediction on SAME measured query CE/adduct/instrument, changing only actual candidate2048fp. Score fixed positive-only mean observed60D query dot predicted candidate diagnostic strength; no learned fusion/multiplier or absent-peak penalty. Equal per-molecule eligible-pair-slot weight by duplicating singleton eligible slots. Report pair contrasts, no dev ranking or activation.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    train = pd.read_parquet(TRAIN)
    pairs = pd.read_parquet(CACHE / "training_rows.parquet")
    if (
        pairs.inchikey14.tolist() != data["keys"]
        or protocol["training_sha256"] != digest(TRAIN)
        or protocol["pair_rows_sha256"] != digest(CACHE / "training_rows.parquet")
    ):
        raise ValueError("Original isolated measured row identity required")
    owner = np.repeat(np.arange(60000), 2)
    rows = train.iloc[
        pairs[["representative_row", "paired_row"]].to_numpy().reshape(-1)
    ].reset_index(drop=True)
    signatures = [energy_signature(e) for e in rows.collision_energy_ev]
    eligible = np.array([e is not None and len(e) == 1 for e in signatures])
    energies = np.array([e[0] for e in signatures if e is not None and len(e) == 1])
    owner = owner[eligible]
    rows = rows.loc[eligible].reset_index(drop=True)
    counts = np.bincount(owner, minlength=60000)
    repeated = np.repeat(np.arange(len(owner)), 2 // counts[owner])
    owner, energies = owner[repeated], energies[repeated]
    rows = rows.iloc[repeated].reset_index(drop=True)
    holdout = heldout_mask(data["keys"])
    use = holdout[owner]
    owner, energies = owner[use], energies[use]
    rows = rows.loc[use].reset_index(drop=True)
    contexts = []
    vocab = json.loads((AUDIT / "private_context_vocabulary.json").read_text())
    for field, classes in vocab.items():
        mapping = {name: i for i, name in enumerate(classes)}
        values = [category(x) for x in rows[field]]
        matrix = np.zeros((len(rows), len(classes) + 1), np.float32)
        matrix[np.arange(len(rows)), [mapping.get(x, len(classes)) for x in values]] = 1
        contexts.append(matrix)
    # JSON vocabulary preserves original adduct/instrument insertion order.
    if list(vocab) != ["adduct", "instrument_type"]:
        raise ValueError("Original context feature order differs")
    contexts = np.concatenate(contexts, axis=1)
    selected = np.concatenate([owner[:, None], hard[owner]], axis=1)
    allowed = admissible_negatives(data["masses"], hard)[owner] & holdout[hard[owner]]
    targets = np.stack(
        [evidence_features(row, True) for row in rows.to_dict("records")]
    )
    arms = {}
    for name, model in models.items():
        scored = np.empty(selected.shape)
        # Each batch queries the SAME observed conditioning for every candidate.
        for start in range(0, len(rows), 128):
            if c.stopped():
                return None
            stop = min(start + 128, len(rows))
            count = stop - start
            predicted = candidate_prediction(
                data["fps"][selected[start:stop].reshape(-1)],
                np.repeat(energies[start:stop], 16),
                np.repeat(contexts[start:stop], 16, axis=0),
                model,
                name == "ce_interaction",
            ).reshape(count, 16, 60)
            scored[start:stop] = (targets[start:stop, None] * predicted).mean(-1)
        arms[name] = contrast_summary(scored[:, 0], scored[:, 1:], allowed)[
            "mass_compatible"
        ]
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "query_molecules": len(set(owner)),
        "equal_weight_query_row_slots": len(rows),
        "both_molecules_heldout_mass_pair_slots": int(allowed.sum()),
        "heldout_targets_used_for_fit": False,
        "train_development_overlap": 0,
        "arms": arms,
        "seconds": time.monotonic() - started,
        "scope": "Train-internal candidate diagnostics only. Each molecule contributes two eligible weighted slots, so pair counts are slot counts not independent molecules. Candidates share query CE/adduct/instrument; fixed mean-dot cannot demonstrate critic-calibrated final ranking value. Reuses383 model choices fixed before dev; no fitting or activation.",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
