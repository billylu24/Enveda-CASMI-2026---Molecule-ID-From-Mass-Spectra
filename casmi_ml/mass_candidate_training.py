"""Matched training of original chemistry on mass-admissible pinned competitors."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np

from casmi_ml import full_gap_candidate_training as base
from casmi_ml.data import write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.mass_candidate_capacity_audit import fit_mass_bonus
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives


def bindings():
    return {
        "mass_training_source_sha256": digest(__file__),
        "mass_optimizer_source_sha256": digest(
            "casmi_ml/mass_candidate_capacity_audit.py"
        ),
        "mass_mask_source_sha256": digest("casmi_ml/strict_mass_negative_mask.py"),
        "pinned_table_sha256": digest(HARD),
        "base_training_source_sha256": digest(base.__file__),
        "train_data_source_sha256": digest("casmi_ml/direct_experiment.py"),
    }


def bind_model(model):
    return {**model, **bindings(), "mass_admissible_denominator": True}


def run(output, mask_chemical_channels=False):
    if not Controller().local_only():
        raise ValueError("Mass-compatible candidate training is local-only")
    output = Path(output)
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    compatible = admissible_negatives(data["masses"], hard)
    freeze(
        output / "mass_training_protocol.json",
        {
            **bindings(),
            "mask_chemical_channels": mask_chemical_channels,
            "rule": "Same original60K/120K/60D/pinned15/critic/encoder/sum2/L2.01/200iter. Exclude only mass-incompatible negatives from training denominator using original35ppm/.006Da mask; always allow positive; all60K equal and no resampling. Deployment candidate windows, critic margin.05 and physical rules unchanged. Matched all-chemistry-zero arm uses the identical denominator.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )

    def fit(cosines, interactions, scale, mask=False, checkpoint=None):
        if mask:
            interactions = np.zeros_like(interactions)
        return fit_mass_bonus(cosines, interactions, scale, compatible, checkpoint)

    with patch.object(base, "fit_bonus", new=fit):
        report = base.run(output, mask_chemical_channels)
    model = bind_model(json.loads((output / "model.json").read_text()))
    write_json(output / "model.json", model)
    report.update(
        model_sha256=digest(output / "model.json"),
        mass_admissible_denominator=True,
        scope="Fixed training-only mass-admissible chemistry fit. Same-mask objective is not accuracy gain. Requires source-bound complete zero/mask controls and both full3340 physical systems; no automatic activation or remote publication.",
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--mask-chemical-channels", action="store_true")
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
