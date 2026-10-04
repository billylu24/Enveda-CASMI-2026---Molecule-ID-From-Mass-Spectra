"""Training-only measured diagnostic stability across original same-ion spectrum pairs."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.data import write_json
from casmi_ml.dino_energy_pairs import energy_signature
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.full_gap_candidate_compatibility import ALL_RULES, evidence_features
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze


def pair_kind(left_energy, right_energy, same_row=False, same_instrument=True):
    if same_row:
        return "duplicated_single_row"
    if not same_instrument:
        return "different_instrument"
    left, right = energy_signature(left_energy), energy_signature(right_energy)
    if left is None or right is None:
        return "missing_energy"
    if len(left) != 1 or len(right) != 1:
        return "multi_energy_signature"
    return "same_single_energy" if left == right else "different_single_energy"


def stability_summary(left, right, lower_energy_first=None):
    left, right = np.asarray(left), np.asarray(right)
    if (
        left.ndim != 2
        or left.shape != right.shape
        or left.shape[1] != 60
        or not np.isfinite(left).all()
        or not np.isfinite(right).all()
        or (left < 0).any()
        or (right < 0).any()
    ):
        raise ValueError("Finite aligned observed60D pair signatures required")
    a, b = left > 0, right > 0
    union = (a | b).sum(1)
    intersection = (a & b).sum(1)
    nonempty = union > 0
    result = {
        "pairs": len(left),
        "pairs_with_evidence_in_either_row": int(nonempty.sum()),
        "pairs_with_evidence_in_both_rows": int((a.any(1) & b.any(1)).sum()),
        "exact_signature_pairs": int((left == right).all(1).sum()),
        "identical_observed_channel_sets": int((a == b).all(1).sum()),
        "mean_jaccard_on_nonempty_pairs": float(
            (intersection[nonempty] / union[nonempty]).mean()
        )
        if nonempty.any()
        else None,
        "mean_squared_strength_difference": float(np.square(left - right).mean())
        if len(left)
        else None,
        "channel_presence_disagreements": (a != b).sum(0).tolist(),
        "channel_mean_absolute_strength_difference": np.abs(left - right)
        .mean(0)
        .tolist()
        if len(left)
        else [None] * 60,
    }
    if lower_energy_first is not None:
        lower = np.asarray(lower_energy_first, dtype=bool)
        if lower.shape != (len(left),):
            raise ValueError("Energy direction must align with complete pairs")
        low = np.where(lower[:, None], left, right)
        high = np.where(lower[:, None], right, left)
        result.update(
            channels_gained_at_higher_energy=((low == 0) & (high > 0)).sum(0).tolist(),
            channels_lost_at_higher_energy=((low > 0) & (high == 0)).sum(0).tolist(),
            channel_mean_higher_minus_lower_strength=(high - low).mean(0).tolist()
            if len(low)
            else [None] * 60,
        )
    return result


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Measured energy audit is local-only")
    output = Path(output)
    started = time.monotonic()
    cache_protocol = json.loads((CACHE / "protocol.json").read_text())
    cache_report = json.loads((CACHE / "report.json").read_text())
    for field, source in {
        "training_sha256": TRAIN,
        "negative_table_sha256": HARD,
        "feature_sha256": "casmi_ml/extended_diagnostic_rules.py",
        "chemistry_sha256": "casmi_ml/chemistry.py",
    }.items():
        if cache_protocol[field] != digest(source):
            raise ValueError("Original isolated measured pair source differs")
    for name, sha in cache_report["private_input_sha256"].items():
        if digest(CACHE / name) != sha:
            raise ValueError("Original complete measured pair cache differs")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "pair_rows_sha256": digest(CACHE / "training_rows.parquet"),
            "evidence_source_sha256": digest(
                "casmi_ml/full_gap_candidate_compatibility.py"
            ),
            "original_feature_source_sha256": digest(
                "casmi_ml/diagnostic_feature_runtime.py"
            ),
            "extended_feature_source_sha256": digest(
                "casmi_ml/extended_diagnostic_rules.py"
            ),
            "energy_source_sha256": digest("casmi_ml/dino_energy_pairs.py"),
            "rule": "All original60000 training same-ion representative/partner pairs; original precise60D diagnostic strength per measured row, no averaging before stability audit. Partition duplicates/different instrument/missing energy/multi-energy signatures/same singleton CE/different singleton CE; never turn merged energies into a scalar CE. Compare observed support Jaccard and strength changes, directional higher/lower singleton CE only. This is observational instrument-stratified evidence, not randomized CE causality or a spectrum simulator. No fitting, dev ranking, tuning, model activation or publication.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    train = pd.read_parquet(TRAIN)
    pairs = pd.read_parquet(CACHE / "training_rows.parquet")
    if len(pairs) != 60000 or cache_report["train_development_overlap"]:
        raise ValueError("Original isolated60000 molecules required")
    ids, partners = [
        pairs[col].to_numpy() for col in ("representative_row", "paired_row")
    ]
    left, right = train.iloc[ids], train.iloc[partners]
    for frame in (left, right):
        if frame.inchikey14.tolist() != pairs.inchikey14.tolist():
            raise ValueError("Measured spectrum identity differs from original pair")
    if not np.array_equal(
        left.adduct.to_numpy(), right.adduct.to_numpy()
    ) or not np.array_equal(
        left.ionization_mode.to_numpy(), right.ionization_mode.to_numpy()
    ):
        raise ValueError("Original same-ion/adduct pairing required")
    positions = np.column_stack([ids, partners]).reshape(-1)
    observed = []
    for start in range(0, len(positions), 1024):
        if c.stopped():
            return None
        observed.extend(
            evidence_features(row, True)
            for row in train.iloc[positions[start : start + 1024]].to_dict("records")
        )
    observed = np.stack(observed).reshape(60000, 2, 60)
    kinds, directions = [], []
    for i, (a, b) in enumerate(zip(left.to_dict("records"), right.to_dict("records"))):
        kind = pair_kind(
            a["collision_energy_ev"],
            b["collision_energy_ev"],
            ids[i] == partners[i],
            a["instrument_type"] == b["instrument_type"],
        )
        kinds.append(kind)
        directions.append(
            energy_signature(a["collision_energy_ev"])[0]
            < energy_signature(b["collision_energy_ev"])[0]
            if kind == "different_single_energy"
            else False
        )
    kinds, directions = np.asarray(kinds), np.asarray(directions)
    arms = {}
    for kind in sorted(set(kinds)):
        use = kinds == kind
        arms[kind] = stability_summary(
            observed[use, 0],
            observed[use, 1],
            directions[use] if kind == "different_single_energy" else None,
        )
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": len(pairs),
        "train_development_overlap": 0,
        "feature_channels": [
            rule.name + ":" + kind
            for rule in ALL_RULES
            for kind in ("ion", "precursor_loss", "peak_loss")
        ],
        "arms": arms,
        "seconds": time.monotonic() - started,
        "scope": "All original isolated training measured pairs, conditional energy stability only. Same/different CE groups can differ in molecular composition, acquisition and preparation; no causal comparison or generalization acceptance. Identical absent signatures are counted separately from evidence Jaccard. No new architecture or deployed model.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
