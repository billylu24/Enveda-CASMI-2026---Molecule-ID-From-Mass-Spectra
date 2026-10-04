"""Source-bound original binary support plus conditional per-row diagnostic strengths."""

import json
from pathlib import Path

import numpy as np

from casmi_ml import full_gap_candidate_compatibility as original
from casmi_ml.data import category
from casmi_ml.dino_energy_pairs import energy_signature
from casmi_ml.metfrag import digest


def score_function(weights, forward, vocab, mask=False):
    weights = np.asarray(weights, np.float64)
    if (
        weights.shape != (120,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() > 2 + 1e-9
    ):
        raise ValueError(
            "Finite nonnegative120 weights within original total2 required"
        )
    weights = weights.astype(np.float32)
    while weights.astype(np.float64).sum() > 2:
        weights = np.nextafter(weights, np.float32(0))
    if list(vocab) != ["adduct", "instrument_type"]:
        raise ValueError("Original condition feature ordering required")
    dimension = 4097 + sum(len(v) + 1 for v in vocab.values())
    if (
        forward["weights"].shape != (dimension, 60)
        or forward["mean_x"].shape != (dimension,)
        or forward["mean_y"].shape != (60,)
        or not all(np.isfinite(x).all() for x in forward.values())
    ):
        raise ValueError("Finite fixed conditional signature ridge required")
    maps = {
        field: {v: i for i, v in enumerate(values)} for field, values in vocab.items()
    }
    matrix = forward["weights"]
    context_weights = matrix[2048:-2049]
    energy_weights = matrix[-2049]
    bias = forward["mean_y"] - forward["mean_x"] @ matrix

    def score(ranker, encoder, group, preprocessing, candidates, fps):
        from casmi_ml.direct_models import score_group

        baseline = score_group(ranker, encoder, group, preprocessing, candidates, fps)
        if mask or not weights.any() or not len(candidates):
            return baseline
        rows = group.to_dict("records")
        chemical = np.stack([original.evidence_features(row, True) for row in rows])
        if not chemical.any():
            return baseline
        support = np.asarray(
            [original.structural_support(s) for s in candidates.normalized_smiles],
            np.float32,
        )
        binary = original.compatibility_features(chemical.mean(0), support)
        conditional = np.zeros((len(candidates), 60), np.float64)
        fps = np.asarray(fps, np.float64)
        projected = fps @ matrix[:2048]
        projected_energy = fps @ matrix[-2048:]
        for observed, row in zip(chemical, rows):
            signature = energy_signature(row.get("collision_energy_ev"))
            if signature is None or len(signature) != 1 or not observed.any():
                continue
            context = []
            for field, values in vocab.items():
                onehot = np.zeros(len(values) + 1)
                onehot[maps[field].get(category(row.get(field)), len(values))] = 1
                context.extend(onehot)
            ce = signature[0] / 100
            predicted = np.clip(
                projected
                + projected_energy * ce
                + np.asarray(context) @ context_weights
                + energy_weights * ce
                + bias,
                0,
                1,
            )
            conditional += observed * predicted / len(rows)
        interactions = np.concatenate([binary, conditional.astype(np.float32)], axis=-1)
        delta = (interactions * weights).sum(-1, dtype=np.float32)
        if delta.shape != baseline.shape or not np.isfinite(delta).all():
            raise ValueError("Conditional score shape/finiteness differs")
        return np.where(delta > 0, baseline + delta, baseline).astype(np.float32)

    return score


def load_model(path, encoder, critic):
    from casmi_ml.research_protocol import TRAIN

    model = json.loads(Path(path).read_text())
    if (
        model.get("bonus_budget") != 2
        or model.get("dimension") != 120
        or not model.get("local_only")
    ):
        raise ValueError("Explicit local120D original total bonus required")
    for field, source in {
        "encoder_sha256": encoder,
        "original_critic_sha256": critic,
        "compatibility_source_sha256": __file__,
        "original_compatibility_source_sha256": "casmi_ml/full_gap_candidate_compatibility.py",
        "capacity_source_sha256": "casmi_ml/energy_conditioned_capacity_audit.py",
        "training_wrapper_source_sha256": "casmi_ml/energy_conditioned_training.py",
        "forward_source_sha256": "casmi_ml/energy_conditioned_diagnostic_audit.py",
        "energy_source_sha256": "casmi_ml/dino_energy_pairs.py",
        "extended_rule_source_sha256": "casmi_ml/extended_diagnostic_rules.py",
        "chemistry_source_sha256": "casmi_ml/chemistry.py",
        "evidence_source_sha256": "casmi_ml/diagnostic_residual.py",
        "feature_source_sha256": "casmi_ml/diagnostic_feature_runtime.py",
        "training_sha256": TRAIN,
    }.items():
        if model.get(field) != digest(source):
            raise ValueError("Conditional model binding differs: " + field)
    assets = {}
    for name in ("forward_model", "context_vocabulary"):
        p = Path(model[name])
        if digest(p) != model[name + "_sha256"]:
            raise ValueError("Conditional model asset differs: " + name)
        assets[name] = p
    with np.load(assets["forward_model"]) as saved:
        forward = {name: saved[name] for name in ("weights", "mean_x", "mean_y")}
    vocab = json.loads(assets["context_vocabulary"].read_text())
    if not isinstance(model.get("mask_chemical_channels"), bool):
        raise TypeError("Explicit chemistry mask required")
    return score_function(
        model["weights"], forward, vocab, model["mask_chemical_channels"]
    ), model
