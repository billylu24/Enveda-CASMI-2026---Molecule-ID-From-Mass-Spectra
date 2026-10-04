"""Training-only local-neighborhood depth audit on original fingerprint aliases."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from casmi_ml.data import fingerprint, write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives

AUDIT = Path("artifacts/research_loop/rounds/0369_training_fingerprint_alias_audit")
RADII = (2, 3, 4, 6, 8)


def radius_signatures(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Valid actual training structure required")
    Chem.RemoveStereochemistry(mol)
    return {
        radius: rdFingerprintGenerator.GetMorganGenerator(radius=radius)
        .GetSparseCountFingerprint(mol)
        .GetNonzeroElements()
        for radius in RADII
    }


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Training radius audit is local-only")
    protocol = json.loads((AUDIT / "protocol.json").read_text())
    for field, source in {
        "source_sha256": "casmi_ml/training_fingerprint_alias_audit.py",
        "training_sha256": TRAIN,
        "hard_table_sha256": HARD,
        "fingerprint_source_sha256": "casmi_ml/data.py",
    }.items():
        if protocol[field] != digest(source):
            raise ValueError("Original complete alias audit source changed")
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "original_audit_sha256": digest(AUDIT / "report.json"),
            "training_sha256": digest(TRAIN),
            "hard_table_sha256": digest(HARD),
            "radii": list(RADII),
            "rule": "Fixed radius2/3/4/6/8 sparse Morgan count comparison ONLY on all originally identical2048bit pinned training pairs. Existing369 complete audit provenance required. No neural fitting, development ranking/labels or new model activation. A distinct sparse signature indicates representation availability, not discriminative spectral evidence or accuracy gain.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    data = train_data()
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    rows = pd.read_parquet(TRAIN, columns=["inchikey14", "normalized_smiles"])
    rows = rows.groupby("inchikey14", sort=True).first()
    if list(rows.index) != data["keys"]:
        raise ValueError("Original training identity order required")
    packed = np.packbits(np.asarray(data["fps"], np.uint8), axis=1)
    pairs = []
    for start in range(0, len(packed), 256):
        if c.stopped():
            return None
        matched = (
            packed[start : start + 256, None] == packed[hard[start : start + 256]]
        ).all(-1)
        for i, slot in np.argwhere(matched):
            i, slot = start + int(i), int(slot)
            pairs.append((i, int(hard[i, slot]), slot))
    old = json.loads((AUDIT / "report.json").read_text())
    if len(pairs) != old["contrasts"]["unrestricted"]["identical_original2048_pairs"]:
        raise ValueError("Complete original identical-pair cohort differs")
    smiles = rows.normalized_smiles.tolist()
    ids = sorted({i for p in pairs for i in p[:2]})
    for i in ids:
        if not np.array_equal(fingerprint(smiles[i]), data["fps"][i]):
            raise ValueError("Original fingerprint structure binding differs")
    signatures = {i: radius_signatures(smiles[i]) for i in ids}
    compatible = admissible_negatives(data["masses"], hard)
    counts = {name: Counter() for name in ("unrestricted", "mass_compatible")}
    for i, j, slot in pairs:
        for name, count in counts.items():
            if name == "mass_compatible" and not compatible[i, slot]:
                continue
            count["pairs"] += 1
            for radius in RADII:
                field = f"radius{radius}_count_separated"
                count[field] += int(
                    signatures[i][radius] != signatures[j][radius]
                )
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "training_molecules": len(packed),
        "train_development_overlap": old["train_development_overlap"],
        "contrasts": {name: dict(value) for name, value in counts.items()},
        "scope": "Availability audit on original fingerprint aliases, no matched neural accuracy, no mechanism proof or whole-pool upper bound. Signature inequality can distinguish this pair; signature equality need not imply actual graph equality. No model replacement.",
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
