"""Training-regularized candidate chemical compatibility under the full cosine-gap ceiling."""

from functools import lru_cache

import numpy as np
from rdkit import Chem

from casmi_ml.chemistry import RULES
from casmi_ml.diagnostic_residual import evidence_features as original_evidence
from casmi_ml.extended_diagnostic_rules import (
    DIM as EXTENDED_DIM,
)
from casmi_ml.extended_diagnostic_rules import (
    RULES as EXTENDED_RULES,
)
from casmi_ml.extended_diagnostic_rules import (
    evidence_features as extended_evidence,
)
from casmi_ml.extended_diagnostic_rules import (
    structural_support as extended_support,
)

DIM = len(RULES) * 3 + EXTENDED_DIM
BASE_RULE_COUNT = len(RULES)
ALL_RULES = RULES + EXTENDED_RULES


def evidence_features(row, precise=True):
    if not precise:
        raise ValueError("Extended candidate hypotheses require precise evidence")
    return np.concatenate([original_evidence(row, True), extended_evidence(row)])


BONUS_BUDGET = 2.0

PATTERNS = tuple(Chem.MolFromSmarts(rule.smarts) for rule in RULES)


@lru_cache(maxsize=100000)
def structural_support(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid candidate for diagnostic support")
    return tuple(
        bool(molecule.HasSubstructMatch(pattern)) for pattern in PATTERNS
    ) + extended_support(smiles)


def compatibility_features(chemical, support):
    chemical, support = (
        np.asarray(chemical, np.float32),
        np.asarray(support, np.float32),
    )
    if chemical.shape[-1] != DIM or support.shape[-1] != len(ALL_RULES):
        raise ValueError("Chemical60 and candidate20 broad rule motifs required")
    if not np.isfinite(chemical).all() or ((chemical < 0) | (chemical > 1)).any():
        raise ValueError("Observed chemistry must be finite within [0,1]")
    if not np.isfinite(support).all() or not np.isin(support, [0, 1]).all():
        raise ValueError("Candidate structural support must be binary")
    return chemical[..., None, :] * np.repeat(support, 3, axis=-1)


def score_function(weights, mask_chemical_channels=False):
    from casmi_ml.direct_models import score_group

    weights = np.asarray(weights, np.float64)
    if weights.shape != (DIM,) or not np.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Nonnegative finite60 candidate diagnostic weights required")

    if float(weights.sum()) > BONUS_BUDGET + 1e-9:
        raise ValueError(
            "Candidate diagnostic weights exceed frozen full cosine-gap bonus budget"
        )

    weights = weights.astype(np.float32)
    # Serialization tofloat32 must not increase the frozen total bonus cap.
    while float(weights.astype(np.float64).sum()) > BONUS_BUDGET:
        weights = np.nextafter(weights, np.float32(0))

    def score(ranker, encoder, group, preprocessing, candidates, fps):
        if getattr(ranker, "uses_chemical_query", False):
            from casmi_ml.diagnostic_query_ranker import score_chemical_group

            baseline = score_chemical_group(
                ranker, encoder, group, preprocessing, candidates, fps
            )
        else:
            baseline = score_group(
                ranker, encoder, group, preprocessing, candidates, fps
            )
        if mask_chemical_channels or not weights.any() or not len(candidates):
            return baseline
        # Average the same measured rows as the frozen critic; chemistry remains
        # conditional on observations, so missing features never subtract score.
        chemical = np.stack(
            [evidence_features(r, True) for r in group.to_dict("records")]
        ).mean(0)
        if not chemical.any():
            return baseline
        support = np.asarray(
            [structural_support(s) for s in candidates.normalized_smiles], np.float32
        )
        delta = (compatibility_features(chemical, support) * weights).sum(
            -1, dtype=np.float32
        )
        if delta.shape != baseline.shape or not np.isfinite(delta).all():
            raise ValueError("Candidate chemistry score shape/finiteness differs")
        return np.where(delta > 0, baseline + delta, baseline).astype(np.float32)

    return score


def load_model(path, encoder, critic):
    """Bind an experimental local model to its actual frozen sources and inputs."""
    import json
    from pathlib import Path

    from casmi_ml.metfrag import digest
    from casmi_ml.research_protocol import TRAIN

    model = json.loads(Path(path).read_text())
    if model.get("bonus_budget") != BONUS_BUDGET or not model.get("local_only"):
        raise ValueError("Frozen local candidate diagnostic budget required")
    bindings = {
        "encoder_sha256": encoder,
        "original_critic_sha256": critic,
        "compatibility_source_sha256": __file__,
        "extended_rule_source_sha256": "casmi_ml/extended_diagnostic_rules.py",
        "chemistry_source_sha256": "casmi_ml/chemistry.py",
        "evidence_source_sha256": "casmi_ml/diagnostic_residual.py",
        "feature_source_sha256": "casmi_ml/diagnostic_feature_runtime.py",
        "training_sha256": TRAIN,
    }
    for name, source in bindings.items():
        if model.get(name) != digest(source):
            raise ValueError("Candidate diagnostic binding differs: " + name)
    if not isinstance(model.get("mask_chemical_channels"), bool):
        raise TypeError("Explicit candidate chemistry mask required")
    score = score_function(model["weights"], model["mask_chemical_channels"])
    return score, model
