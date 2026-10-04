"""Original inference score with explicit mass-admissible training provenance."""

import json
from pathlib import Path

from casmi_ml.full_gap_candidate_compatibility import load_model as original_load
from casmi_ml.mass_candidate_training import bindings


def load_model(path, encoder, critic):
    model = json.loads(Path(path).read_text())
    if model.get("mass_admissible_denominator") is not True:
        raise ValueError("Explicit mass-admissible training provenance required")
    for name, expected in bindings().items():
        if model.get(name) != expected:
            raise ValueError(
                "Mass-compatible candidate training binding differs: " + name
            )
    # Runtime features, mass windows and physical ranking remain the original ones.
    return original_load(path, encoder, critic)
