"""Training-only chemical discriminability against actual mass-compatible pinned negatives."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.candidate_diagnostic_compatibility import load_model, structural_support
from casmi_ml.chemistry import RULES
from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import KINDS, evidence_features
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives
from casmi_ml.training_query_pairs import paired_rows

MODEL = Path(
    "artifacts/research_loop/rounds/0300_candidate_compatibility_chemical/model.json"
)


def run(output):
    if not Controller().local_only():
        raise ValueError("Candidate diagnostic audit is local-only")
    output = Path(output)
    _, model = load_model(MODEL, ENCODER, CRITIC)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "pinned_table_sha256": digest(HARD),
            "model_sha256": digest(MODEL),
            "rule": "All original60K representatives+same-ion pairs; compare observed chemistry times broad actual candidate SMARTS against original pinned15 negatives, unrestricted vs35ppm/floor.006Da training-mass compatibility. Aggregate-only, no dev data or parameter fitting.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    train = pd.read_parquet(TRAIN)
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    groups = train.groupby("inchikey14", sort=True).indices
    if list(groups) != data["keys"]:
        raise ValueError("Training chemical and negative identities differ")
    rng = np.random.default_rng(42)
    ids = np.array([rng.choice(v) for v in groups.values()])
    pair = paired_rows(train, ids)
    selected_rows = np.column_stack([ids, pair]).reshape(-1)
    chemical_rows = []
    for start in range(0, len(selected_rows), 1024):
        if Controller().stopped():
            return None
        chemical_rows.extend(
            evidence_features(r, True)
            for r in train.iloc[selected_rows[start : start + 1024]].to_dict("records")
        )
    chemical = np.stack(chemical_rows).reshape(len(ids), 2, 24).mean(1)
    support = np.repeat(
        np.asarray(
            [structural_support(s) for s in train.iloc[ids].normalized_smiles],
            np.float32,
        ),
        3,
        axis=-1,
    )
    compatible = admissible_negatives(data["masses"], hard)
    differences = chemical[:, None] * (support[:, None] - support[hard])
    weighted = differences @ np.asarray(model["weights"], np.float64)
    arms = {}
    for name, mask in [
        ("unrestricted", np.ones_like(compatible)),
        ("mass_compatible", compatible),
    ]:
        contrast = differences[mask]
        delta = weighted[mask]
        arms[name] = {
            "pairs": int(mask.sum()),
            "molecules_with_negative": int(mask.any(1).sum()),
            "pairs_with_any_candidate_motif_difference": int(
                (np.abs(contrast) > 0).any(1).sum()
            ),
            "pairs_with_trained_bonus_difference": int((np.abs(delta) > 1e-9).sum()),
            "pairs_bonus_favors_training_positive": int((delta > 1e-9).sum()),
            "pairs_bonus_favors_training_negative": int((delta < -1e-9).sum()),
            "mean_positive_minus_negative_bonus": float(delta.mean())
            if len(delta)
            else None,
            "channels": [
                {
                    "channel": rule.name + ":" + kind,
                    "weight": model["weights"][i * 3 + j],
                    "positive_support_advantage_pairs": int(
                        (contrast[:, i * 3 + j] > 0).sum()
                    ),
                    "negative_support_advantage_pairs": int(
                        (contrast[:, i * 3 + j] < 0).sum()
                    ),
                    "mean_interaction_positive_minus_negative": float(
                        contrast[:, i * 3 + j].mean()
                    )
                    if len(contrast)
                    else None,
                }
                for i, rule in enumerate(RULES)
                for j, kind in enumerate(KINDS)
            ],
        }
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": len(ids),
        "training_slots": len(selected_rows),
        "arms": arms,
        "scope": "Training-only chemical candidate discrimination; molecular mass is permitted training structure property, not test truth. Pinned compatible subset is not a complete deployed competitor pool; no accuracy or generalization claim.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
