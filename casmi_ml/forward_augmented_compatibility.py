"""Source-bound original chemistry plus predicted diagnostic candidate support."""

import json
from pathlib import Path

import numpy as np

from casmi_ml import full_gap_candidate_compatibility as original
from casmi_ml.metfrag import digest

DIM = 120


def score_function(weights, forward, mask=False):
    weights = np.asarray(weights, np.float64)
    if (
        weights.shape != (DIM,)
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
    if (
        forward["weights"].shape != (2048, 60)
        or forward["mean_x"].shape != (2048,)
        or forward["mean_y"].shape != (60,)
        or not all(np.isfinite(v).all() for v in forward.values())
    ):
        raise ValueError("Finite original2048-to-diagnostic60 ridge model required")

    def score(ranker, encoder, group, preprocessing, candidates, fps):
        from casmi_ml.direct_models import score_group

        baseline = score_group(ranker, encoder, group, preprocessing, candidates, fps)
        if mask or not weights.any() or not len(candidates):
            return baseline
        chemical = np.stack(
            [original.evidence_features(r, True) for r in group.to_dict("records")]
        ).mean(0)
        if not chemical.any():
            return baseline
        support = np.asarray(
            [original.structural_support(s) for s in candidates.normalized_smiles],
            np.float32,
        )
        predicted = np.clip(
            (np.asarray(fps, np.float64) - forward["mean_x"]) @ forward["weights"]
            + forward["mean_y"],
            0,
            1,
        ).astype(np.float32)
        interactions = np.concatenate(
            [
                original.compatibility_features(chemical, support),
                chemical[None, :] * predicted,
            ],
            axis=-1,
        )
        delta = (interactions * weights).sum(-1, dtype=np.float32)
        if delta.shape != baseline.shape or not np.isfinite(delta).all():
            raise ValueError("Forward augmented score shape/finiteness differs")
        return np.where(delta > 0, baseline + delta, baseline).astype(np.float32)

    return score


def load_model(path, encoder, critic):
    from casmi_ml.research_protocol import TRAIN

    model = json.loads(Path(path).read_text())
    if (
        model.get("bonus_budget") != 2
        or not model.get("local_only")
        or model.get("dimension") != DIM
    ):
        raise ValueError("Explicit local120D original total bonus bound required")
    for field, source in {
        "encoder_sha256": encoder,
        "original_critic_sha256": critic,
        "compatibility_source_sha256": __file__,
        "original_compatibility_source_sha256": "casmi_ml/full_gap_candidate_compatibility.py",
        "forward_source_sha256": "casmi_ml/forward_diagnostic_training_audit.py",
        "training_wrapper_source_sha256": "casmi_ml/forward_augmented_training.py",
        "extended_rule_source_sha256": "casmi_ml/extended_diagnostic_rules.py",
        "chemistry_source_sha256": "casmi_ml/chemistry.py",
        "evidence_source_sha256": "casmi_ml/diagnostic_residual.py",
        "feature_source_sha256": "casmi_ml/diagnostic_feature_runtime.py",
        "training_sha256": TRAIN,
    }.items():
        if model.get(field) != digest(source):
            raise ValueError("Forward augmented binding differs: " + field)
    forward = Path(model["forward_model"])
    if digest(forward) != model["forward_model_sha256"]:
        raise ValueError("Fixed training-only forward model changed")
    with np.load(forward) as saved:
        values = {name: saved[name] for name in ("weights", "mean_x", "mean_y")}
    if not isinstance(model.get("mask_chemical_channels"), bool):
        raise TypeError("Explicit chemical identity mask required")
    return score_function(
        model["weights"], values, model["mask_chemical_channels"]
    ), model
