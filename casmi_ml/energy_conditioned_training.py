"""Matched original120D conditioned capacity recipe for chemical/masked full systems."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

from casmi_ml import energy_conditioned_capacity_audit as capacity
from casmi_ml.data import write_json
from casmi_ml.full_gap_candidate_bonus import fit_bonus
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

AUDIT = Path("artifacts/research_loop/rounds/0386_energy_conditioned_capacity_audit")


def bind_model(model, forward, vocabulary):
    model.update(
        weights=[0.0] * 120 if not any(model["weights"]) else model["weights"],
        dimension=120,
        compatibility_source_sha256=digest(
            "casmi_ml/energy_conditioned_compatibility.py"
        ),
        original_compatibility_source_sha256=digest(
            "casmi_ml/full_gap_candidate_compatibility.py"
        ),
        capacity_source_sha256=digest(capacity.__file__),
        training_wrapper_source_sha256=digest(__file__),
        forward_source_sha256=digest("casmi_ml/energy_conditioned_diagnostic_audit.py"),
        energy_source_sha256=digest("casmi_ml/dino_energy_pairs.py"),
        forward_model=str(Path(forward).resolve()),
        forward_model_sha256=digest(forward),
        context_vocabulary=str(Path(vocabulary).resolve()),
        context_vocabulary_sha256=digest(vocabulary),
    )
    return model


def run(output, mask_chemical_channels=False):
    output = Path(output)
    probe = json.loads((AUDIT / "report.json").read_text())
    protocol = json.loads((AUDIT / "conditional_capacity_protocol.json").read_text())
    if (
        protocol["source_sha256"] != digest(capacity.__file__)
        or probe["capacity_probe"]["same_objective_gain"] <= 0
    ):
        raise ValueError(
            "Completed current-source positive conditional capacity required"
        )
    freeze(
        output / "conditional_training_protocol.json",
        {
            "source_sha256": digest(__file__),
            "capacity_source_sha256": digest(capacity.__file__),
            "capacity_report_sha256": digest(AUDIT / "report.json"),
            "score_source_sha256": digest(
                "casmi_ml/energy_conditioned_compatibility.py"
            ),
            "mask_chemical_channels": mask_chemical_channels,
            "rule": "386 exact original60000/120000 same-ion paired conditional ridge and120D bonus recipe; sum2/L2.01/200iter/critic/encoder/pinned15 and originalmargin.05 unchanged. This wrapper takes newly fitted120D weights instead of returned original60D baseline; masked arm performs identical conditional feature/model preparation then masks ALL bonus channels. Both complete3340 physical systems and exact controls before evaluation. No dev hyperparameters or remote publication.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    captured = []

    def fitter(cosines, interactions, scale, mask=False, checkpoint=None):
        weights, history = fit_bonus(
            cosines, interactions, scale, mask_chemical_channels, checkpoint
        )
        if len(weights) == 120:
            captured.append((weights, history))
        return weights, history

    with patch.object(capacity, "fit_bonus", new=fitter):
        report = capacity.run(output)
    if len(captured) != 1:
        raise ValueError("One original120D capacity fit required")
    model = json.loads((output / "model.json").read_text())
    model.update(
        weights=captured[0][0].tolist(),
        mask_chemical_channels=mask_chemical_channels,
        objective=captured[0][1],
    )
    model = bind_model(
        model,
        output / "private_conditional_signature_model.npz",
        output / "private_context_vocabulary.json",
    )
    write_json(output / "model.json", model)
    report.update(
        model_sha256=digest(output / "model.json"),
        dimension=120,
        mask_chemical_channels=mask_chemical_channels,
        history=captured[0][1],
        scope="Fixed train-only conditional120D chemistry/mask preparation; require full3340 zero and mask controls and complete physical systems before scientific ranking evaluation. No publication or activation.",
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
