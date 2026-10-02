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
    deployment_path = directory / "deployment.json"
    deployment = (
        json.loads(deployment_path.read_text()) if deployment_path.exists() else {}
    )
    if deployment:
        from casmi_ml.metfrag import digest

        if digest(directory / "protocol.json") != deployment["protocol_sha256"]:
            raise ValueError("Selected development protocol changed")
        protocol = {**protocol, **deployment}
    prefix, slots = 5, 5
    frequency_weight = 0.0
    if protocol.get("candidate_statistics"):
        decision = json.loads((directory / "decision.json").read_text())
        frequency_weight = protocol["variants"][decision["winner"]["variant"]]
    if protocol.get("slot_ablation"):
        decision = json.loads((directory / "decision.json").read_text())
        _, prefix, slots = decision["winner"]["variant"].split("_")
        prefix, slots = int(prefix), int(slots)
    checkpoint = Path(
        protocol.get("generator_checkpoint", ROOT / "generation/smiles_42/model.pt")
    )
    if "generator_checkpoint" in protocol:
        prefix, slots = protocol["prefix"], protocol["slots"]
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
    from casmi_ml.metfrag import digest

    suffix = (
        ""
        if checkpoint.resolve() == (ROOT / "generation/smiles_42/model.pt").resolve()
        else "_" + digest(checkpoint)[:12]
    )
    generated_path = (
        ROOT / "generation" / f"researchdev_samples128_limitall_stable_v2{suffix}.json"
    )
    if protocol.get("candidate_statistics") or deployment:
        generated_path = Path(protocol["generated_path"])
    generate = json.loads(generated_path.read_text())
    if deployment and deployment.get("calibrated"):
        from casmi_ml.generated_score_combination import combined_order
        from casmi_ml.metfrag import digest

        scores_path = Path(deployment["critic_scores_path"])
        if digest(scores_path) != deployment["critic_scores_sha256"]:
            raise ValueError("Frozen critic development scores changed")
        scores = json.loads(scores_path.read_text())
        generated = {
            r["key"]: combined_order(
                r["candidates"],
                scores.get(r["key"], {}).get("fingerprint", {}),
                (1, 0, 0.5),
            )
            for r in generate
        }
    elif frequency_weight:
        from casmi_ml.generation_frequency_ranking import ranked_candidates

        generated = {
            r["key"]: ranked_candidates(r["candidates"], frequency_weight)
            for r in generate
        }
    else:
        generated = {r["key"]: [c["key"] for c in r["candidates"]] for r in generate}
    open_protected = protocol.get("open_protected", False)
    by_branch = {"high": [], "low": [], "expanded": []}
    for row in rows:
        key = row["key"]
        if confidence[key] >= 0.5 and not open_protected:
            continue
        branch = protects_reference(
            row["variants"]["baseline"]["ranking"], observed, confidence[key]
        )
        route = "high" if confidence[key] >= 0.5 else "low" if branch else "expanded"
        if len(by_branch[route]) < limit and (not branch or generated[key]):
            by_branch[route].append(row)
    chosen = by_branch["high"] + by_branch["low"] + by_branch["expanded"]
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
    frame = frame.drop(
        columns=[
            c
            for c in [
                "inchikey14",
                "normalized_smiles",
                "molecular_formula",
                "fingerprint",
            ]
            if c in frame
        ]
    )
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
            {
                "molecule_id": key,
                "protected": confidence[key] >= 0.5,
                "generation_allowed": allowed,
            }
        )
        expected[key] = (
            insert_generated(rank, generated[key], prefix, slots) if allowed else rank
        )[:25]
    write_json(output / "full.json", full)
    pd.DataFrame(base).to_csv(output / "base.csv", index=False)
    pd.DataFrame(routing).to_csv(output / "routing.csv", index=False)
    predict(
        checkpoint,
        output / "test.parquet",
        output / "base.csv",
        output / "submission.csv",
        routing_csv=output / "routing.csv",
        encoder_path=ENCODER,
        prefix=prefix,
        slots=slots,
        full_rankings=output / "full.json",
        open_protected=open_protected,
        frequency_weight=frequency_weight,
        token_length_exponent=1.0 if deployment.get("calibrated") else 0.0,
        critic_checkpoint=deployment.get("critic_checkpoint"),
        critic_weight=0.5 if deployment.get("calibrated") else 0.0,
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
        "generation_branch": len(by_branch["high"]) + len(by_branch["low"]),
        "high_confidence_branch": len(by_branch["high"]),
        "expanded_branch": len(by_branch["expanded"]),
        "open_protected": open_protected,
        "prefix": prefix,
        "slots": slots,
        "frequency_weight": frequency_weight,
        "token_length_exponent": 1.0 if deployment.get("calibrated") else 0.0,
        "critic_weight": 0.5 if deployment.get("calibrated") else 0.0,
        "critic_sha256": deployment.get("critic_sha256"),
        "generator_sha256": digest(checkpoint),
        "samples_sha256": digest(generated_path),
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
