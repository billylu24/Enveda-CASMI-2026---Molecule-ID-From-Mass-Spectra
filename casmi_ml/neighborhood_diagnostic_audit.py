"""Fixed molecular-holdout diagnostic prediction with deeper count fingerprints."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from casmi_ml import forward_diagnostic_training_audit as base
from casmi_ml.candidate_chemistry_specificity_audit import contrast_summary
from casmi_ml.data import write_json
from casmi_ml.diagnostic_bonus_generalization_audit import molecular_bootstrap_gain
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD
from casmi_ml.energy_conditioned_precision_audit import bootstrap_summary
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives

PREVIOUS = Path("artifacts/research_loop/rounds/0372_forward_diagnostic_training_audit")


def count_features(smiles, radius):
    if radius not in (2, 3):
        raise ValueError("Only prespecified radius2/3 arms allowed")
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None or not mol.GetNumAtoms():
        raise ValueError("Valid nonempty training structure required")
    Chem.RemoveStereochemistry(mol)
    values = rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=2048, includeChirality=False
    ).GetCountFingerprintAsNumPy(mol)
    return np.log1p(values.astype(np.float64)).astype(np.float32)


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Local-only training diagnostic required")
    output = Path(output)
    prior = json.loads((PREVIOUS / "protocol.json").read_text())
    expected = json.loads((PREVIOUS / "report.json").read_text())
    if prior["source_sha256"] != digest(base.__file__):
        raise ValueError("Unchanged original372 audit required")
    data = train_data()
    hard = np.load(HARD)
    holdout = base.heldout_mask(data["keys"])
    rows = pd.read_parquet(TRAIN, columns=["inchikey14", "normalized_smiles"])
    smiles = rows.groupby("inchikey14", sort=True).first()
    if list(smiles.index) != data["keys"]:
        raise ValueError("Original60000 structure order required")
    captured, predictions = {}, {}
    original_fit = base.fit_ridge

    def fit(features, targets, fit_rows, **kwargs):
        values, model = original_fit(features, targets, fit_rows, **kwargs)
        if features.shape[1] == 2048:
            if not np.array_equal(features, data["fps"]) or not np.array_equal(
                fit_rows, ~holdout
            ):
                raise ValueError("Exact original binary2048/split required")
            captured["targets"] = targets
            predictions["radius2_binary"] = values
            for radius in (2, 3):
                if c.stopped():
                    raise RuntimeError("User stop observed")
                name = f"radius{radius}_log_count"
                print("Building/fitting fixed " + name, flush=True)
                encoded = np.stack(
                    [count_features(s, radius) for s in smiles.normalized_smiles]
                )
                predictions[name], fitted = original_fit(
                    encoded, targets, fit_rows, **kwargs
                )
                np.savez_compressed(output / ("private_" + name + ".npz"), **fitted)
                del encoded, fitted
        return values, model

    def bound_freeze(path, protocol):
        protocol.update(
            source_sha256=digest(__file__),
            original_audit_source_sha256=digest(base.__file__),
            original_report_sha256=digest(PREVIOUS / "report.json"),
            bootstrap_source_sha256=digest(
                "casmi_ml/energy_conditioned_precision_audit.py"
            ),
            loss_bootstrap_source_sha256=digest(
                "casmi_ml/diagnostic_bonus_generalization_audit.py"
            ),
            rule=protocol["rule"] + " Reconstruct original372 complete arms exactly. "
            "Additional arms fixed2048 hashed Morgan radius2/radius3 log1p(count), "
            "same .01 ridge/unpenalized intercept/clamp0..1 and molecular split. "
            "Matched radius2count vs radius3count comparison separates neighborhood "
            "depth under the same count transform/dimension. No hyperparameter "
            "search or development ranking. Whole12006 heldout query molecular "
            "bootstrap includes zero eligible-negative molecules. No deployment.",
        )
        freeze(path, protocol)

    with (
        patch.object(base, "fit_ridge", new=fit),
        patch.object(base, "freeze", new=bound_freeze),
    ):
        report = base.run(output)
    if report is None:
        return None
    if report["arms"] != expected["arms"] or len(predictions) != 3:
        raise ValueError("All original372 summary arms must reconstruct exactly")
    targets = captured["targets"]
    query = targets[holdout]
    allowed = admissible_negatives(data["masses"], hard)[holdout]
    allowed &= holdout[hard[holdout]]
    arms, margins, losses = {}, {}, {}
    for name, values in predictions.items():
        positive = (query * values[holdout]).mean(1)
        negative = (query[:, None] * values[hard[holdout]]).mean(-1)
        margins[name] = positive[:, None] - negative
        losses[name] = np.square(values[holdout] - query).mean(1)
        arms[name] = {
            "heldout_mse": float(losses[name].mean()),
            "both_heldout_mass_contrasts": contrast_summary(
                positive, negative, allowed
            )["mass_compatible"],
        }
    comparisons = {}
    for left, right in (
        ("radius2_binary", "radius2_log_count"),
        ("radius2_binary", "radius3_log_count"),
        ("radius2_log_count", "radius3_log_count"),
    ):
        comparisons[left + "_to_" + right] = {
            "mse_reduction": molecular_bootstrap_gain(losses[left], losses[right]),
            "pair_margin_gain": bootstrap_summary(
                ((margins[right] - margins[left]) * allowed).sum(1), allowed.sum(1)
            ),
        }
    report.update(
        original372_all_summary_arms_exact=True,
        neighborhood_arms=arms,
        neighborhood_comparisons=comparisons,
        heldout_molecules_without_eligible_negatives=int((~allowed.any(1)).sum()),
        scope="Fixed training-internal molecular holdout diagnostic prediction "
        "and positive-only pair contrast, no critic fitting or calibrated ranking. "
        "Radius/count/hashed representation effects do not establish fragmentation "
        "mechanisms or full-system accuracy. Repeatedly inspected training split, "
        "no independent end-to-end acceptance, deployment or publication.",
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
