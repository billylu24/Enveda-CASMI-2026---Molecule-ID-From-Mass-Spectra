"""Frozen critic with same60D evidence refined by three aromatic original-H ion compositions."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from casmi_ml.aromatic_h_capacity_audit import refine_interactions
from casmi_ml.direct_models import score_group
from casmi_ml.fragment_hydrogen_audit import component_hydrogen_offsets
from casmi_ml.full_gap_candidate_compatibility import (
    BONUS_BUDGET,
    DIM,
    compatibility_features,
    evidence_features,
    structural_support,
)
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import TRAIN


@lru_cache(maxsize=100000)
def cached_connected_support(smiles):
    return tuple(0 in offsets for offsets in component_hydrogen_offsets(smiles))


def score_function(weights, mask_chemical_channels=False):
    weights = np.asarray(weights, np.float64)
    if (
        weights.shape != (DIM,)
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() > BONUS_BUDGET + 1e-9
    ):
        raise ValueError(
            "Finite60 nonnegative weights with fixed2 total budget required"
        )
    weights = weights.astype(np.float32)
    while float(weights.astype(np.float64).sum()) > BONUS_BUDGET:
        weights = np.nextafter(weights, np.float32(0))

    def score(ranker, encoder, group, preprocessing, candidates, fps):
        if getattr(ranker, "uses_chemical_query", False):
            raise ValueError("Connected compatibility requires original frozen critic")
        baseline = score_group(ranker, encoder, group, preprocessing, candidates, fps)
        if mask_chemical_channels or not weights.any() or not len(candidates):
            return baseline
        chemical = np.stack(
            [evidence_features(r, True) for r in group.to_dict("records")]
        ).mean(0)
        if not chemical.any():
            return baseline
        support = np.asarray(
            [structural_support(s) for s in candidates.normalized_smiles], np.float32
        )
        connected = np.asarray(
            [cached_connected_support(s) for s in candidates.normalized_smiles], bool
        )
        interactions = refine_interactions(
            compatibility_features(chemical, support), connected
        )
        delta = (interactions * weights).sum(-1, dtype=np.float32)
        if delta.shape != baseline.shape or not np.isfinite(delta).all():
            raise ValueError("Connected score shape/finiteness differs")
        return np.where(delta > 0, baseline + delta, baseline).astype(np.float32)

    return score


def load_model(path, encoder, critic):
    model = json.loads(Path(path).read_text())
    if model.get("bonus_budget") != BONUS_BUDGET or not model.get("local_only"):
        raise ValueError("Fixed local60D/2.0 model required")
    sources = {
        "encoder_sha256": encoder,
        "original_critic_sha256": critic,
        "compatibility_source_sha256": __file__,
        "original_compatibility_source_sha256": "casmi_ml/full_gap_candidate_compatibility.py",
        "hydrogen_source_sha256": "casmi_ml/fragment_hydrogen_audit.py",
        "refinement_source_sha256": "casmi_ml/aromatic_h_capacity_audit.py",
        "extended_rule_source_sha256": "casmi_ml/extended_diagnostic_rules.py",
        "chemistry_source_sha256": "casmi_ml/chemistry.py",
        "evidence_source_sha256": "casmi_ml/diagnostic_residual.py",
        "feature_source_sha256": "casmi_ml/diagnostic_feature_runtime.py",
        "training_sha256": TRAIN,
    }
    for field, source in sources.items():
        if model.get(field) != digest(source):
            raise ValueError("Frozen connected model binding differs: " + field)
    if not isinstance(model.get("mask_chemical_channels"), bool):
        raise TypeError("Explicit chemical identity mask required")
    return score_function(model["weights"], model["mask_chemical_channels"]), model
