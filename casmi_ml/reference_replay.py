"""Actual generation handoff replay for both reference-protected and expanded branches."""

import argparse
import json
from pathlib import Path

import pandas as pd
from rdkit import Chem

from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.coverage_experiment import EXTERNAL
from casmi_ml.data import write_json
from casmi_ml.generation_inference import predict
from casmi_ml.generation_slots import insert_generated
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_protocol import ENCODER, ROOT


def run(directory, limit=25):
    directory = Path(directory)
    output = directory / "replay"
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((directory / "protocol.json").read_text())
    source = Path(protocol["source_directory"])
    rows = json.loads((source / "unknown_records.json").read_text())
    confidence = {
        r["key"]: r["confidence"]
        for r in json.loads((ROOT / "researchdev_unknown_chemical.json").read_text())
    }
    observed = set(
        pd.read_parquet(
            "artifacts/research_loop/rounds/0001_mass_v2/reference/unknown/rows.parquet",
            columns=["inchikey14"],
        ).inchikey14
    )
    generate = json.loads(
        (ROOT / "generation/researchdev_samples128_limitall_stable_v2.json").read_text()
    )
    generated = {r["key"]: [c["key"] for c in r["candidates"]] for r in generate}
    by_branch = {True: [], False: []}
    for row in rows:
        key = row["key"]
        if confidence[key] >= 0.5:
            continue
        branch = protects_reference(
            row["variants"]["baseline"]["ranking"], observed, confidence[key]
        )
        if len(by_branch[branch]) < limit and (not branch or generated[key]):
            by_branch[branch].append(row)
    chosen = by_branch[True] + by_branch[False]
    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    external = pd.read_parquet(EXTERNAL)
    lookup.update(
        {
            r.inchikey14: r.normalized_smiles
            for r in external.itertuples()
            if r.inchikey14 not in lookup
        }
    )
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    keys = [r["key"] for r in chosen]
    frame = frame[frame.inchikey14.isin(keys)].copy()
    frame["molecule_id"] = frame.inchikey14
    frame = frame.drop(columns=["inchikey14", "normalized_smiles", "molecular_formula"])
    frame.to_parquet(output / "test.parquet")
    full, base, routing, expected = [], [], [], {}
    for row in chosen:
        key = row["key"]
        allowed = protects_reference(
            row["variants"]["baseline"]["ranking"], observed, confidence[key]
        )
        rank = row["variants"]["baseline" if allowed else "expansion_1"]["ranking"]
        smiles = [lookup[k] for k in rank]
        full.append({"molecule_id": key, "smiles": smiles})
        base.append({"molecule_id": key, "smiles": ";".join(smiles[:25])})
        routing.append(
            {"molecule_id": key, "protected": False, "generation_allowed": allowed}
        )
        expected[key] = (
            insert_generated(rank, generated[key], 5, 5) if allowed else rank
        )[:25]
    write_json(output / "full.json", full)
    pd.DataFrame(base).to_csv(output / "base.csv", index=False)
    pd.DataFrame(routing).to_csv(output / "routing.csv", index=False)
    predict(
        ROOT / "generation/smiles_42/model.pt",
        output / "test.parquet",
        output / "base.csv",
        output / "submission.csv",
        routing_csv=output / "routing.csv",
        encoder_path=ENCODER,
        prefix=5,
        slots=5,
        full_rankings=output / "full.json",
    )
    actual = pd.read_csv(output / "submission.csv").set_index("molecule_id")
    matches = sum(
        [
            Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]
            for s in actual.loc[k, "smiles"].split(";")
        ]
        == v
        for k, v in expected.items()
    )
    result = {
        "valid": matches == len(chosen),
        "molecules": len(chosen),
        "complete_top25_matches": matches,
        "generation_branch": len(by_branch[True]),
        "expanded_branch": len(by_branch[False]),
        "scope": "Actual unlabeled generation handoff for both routing branches; expansion CPU/MetFrag replay separately verified",
    }
    write_json(directory / "replay.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    print(run(a.directory))


if __name__ == "__main__":
    main()
