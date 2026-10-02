"""Actual low-confidence, unlabeled inference replay against frozen development rows."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.generation_inference import predict
from casmi_ml.generation_slots import insert_generated
from casmi_ml.research_protocol import ENCODER, ROOT


def run(
    generated,
    incumbent,
    output,
    limit=25,
    prefix=5,
    slots=3,
    checkpoint=None,
    open_protected=False,
):
    generated, incumbent, output = Path(generated), Path(incumbent), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    confidence = {
        r["key"]: r["confidence"]
        for r in json.loads((ROOT / "researchdev_unknown_chemical.json").read_text())
    }
    all_samples = json.loads(generated.read_text())
    samples = [
        r
        for r in all_samples
        if r["candidates"]
        and (
            confidence[r["key"]] >= 0.5
            if open_protected
            else confidence[r["key"]] < 0.5
        )
    ][:limit]
    if len(samples) < limit:
        used = {r["key"] for r in samples}
        samples.extend(r for r in all_samples if r["key"] not in used)
        samples = samples[:limit]
    keys = [r["key"] for r in samples]
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    frame = frame[frame.inchikey14.isin(keys)].copy()
    frame["molecule_id"] = frame.inchikey14
    # Competition-like inputs contain no truth structure/formula/key.
    frame = frame.drop(
        columns=[
            c
            for c in ["inchikey14", "normalized_smiles", "molecular_formula"]
            if c in frame
        ]
    )
    frame.to_parquet(output / "test.parquet")
    retrieval = {
        r["key"]: r["variants"]["charge_aware_union"]["ranking"]
        for r in json.loads((incumbent / "unknown_records.json").read_text())
    }
    confidence = {
        r["key"]: r["confidence"]
        for r in json.loads((ROOT / "researchdev_unknown_chemical.json").read_text())
    }
    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    pd.DataFrame(
        [
            {"molecule_id": k, "smiles": ";".join(lookup[x] for x in retrieval[k][:25])}
            for k in keys
        ]
    ).to_csv(output / "base.csv", index=False)
    pd.DataFrame(
        [{"molecule_id": k, "protected": confidence[k] >= 0.5} for k in keys]
    ).to_csv(output / "routing.csv", index=False)
    write_json(
        output / "full.json",
        [{"molecule_id": k, "smiles": [lookup[x] for x in retrieval[k]]} for k in keys],
    )
    predict(
        checkpoint or ROOT / "generation/smiles_42/model.pt",
        output / "test.parquet",
        output / "base.csv",
        output / "submission.csv",
        routing_csv=output / "routing.csv",
        encoder_path=ENCODER,
        prefix=prefix,
        slots=slots,
        full_rankings=output / "full.json",
        open_protected=open_protected,
    )
    actual = pd.read_csv(output / "submission.csv").set_index("molecule_id")
    matches = 0
    for row in samples:
        k = row["key"]
        expected = (
            retrieval[k]
            if confidence[k] >= 0.5 and not open_protected
            else insert_generated(
                retrieval[k], [c["key"] for c in row["candidates"]], prefix, slots
            )
        )
        predicted = [
            Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]
            for s in actual.loc[k, "smiles"].split(";")
        ]
        matches += predicted == expected[:25]
    result = {
        "molecules": len(keys),
        "open_protected": open_protected,
        "protected_queries": sum(confidence[k] >= 0.5 for k in keys),
        "prefix": prefix,
        "slots": slots,
        "full_top25_matches": matches,
        "valid": matches == len(keys),
        "scope": "Implementation replay of development queries, no new statistical evidence",
    }
    write_json(output / "verification.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--generated", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--limit", type=int, default=25)
    p.add_argument("--prefix", type=int, default=5)
    p.add_argument("--slots", type=int, default=3)
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--open-protected", action="store_true")
    a = p.parse_args()
    print(
        run(
            a.generated, a.incumbent, a.output, a.limit, a.prefix, a.slots, a.checkpoint
        )
    )


if __name__ == "__main__":
    main()
