"""Independent PubChemLite candidate expansion under frozen spectrum scoring."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi_ml import ablation
from casmi_ml.ablation import sha
from casmi_ml.candidate_catalog import expanded_pool
from casmi_ml.data import write_json
from casmi_ml.direct_experiment import single_query
from casmi_ml.experiment import bootstrap_difference
from casmi_ml.inference import group_probability
from casmi_ml.ranking import (
    CandidateIndex,
    ReferenceIndex,
    baseline_rank,
    build_candidates,
    center_mass,
    metrics,
    neural_rank,
    rrf,
)
from casmi_ml.scale_experiment import COCONUT, OLD, sample_keys
from casmi_ml.scale_experiment import ROOT as SCALE
from casmi_ml.secondary_inference import load_deployment_checkpoint, low_confidence_rank
from casmi_ml.training import configure

ROOT = Path("artifacts/candidate_expansion_20260929")
PREVIOUS = Path("artifacts/direct_rank_20260929")
EXTERNAL = Path("external/pubchemlite")
CATALOG = EXTERNAL / "structures.parquet"
ENCODER = SCALE / "runs/wide_enhanced_60000/model.pt"


def prepare_catalog():
    if CATALOG.exists():
        return
    path = EXTERNAL / "PubChemLite_exposomics_20260925.csv"
    meta = json.loads((EXTERNAL / "zenodo_record.json").read_text())
    assert (
        "md5:" + hashlib.md5(path.read_bytes()).hexdigest()
        == meta["files"][0]["checksum"]
    )
    raw = pd.read_csv(
        path,
        usecols=[
            "Identifier",
            "FirstBlock",
            "SMILES",
            "MolecularFormula",
            "MonoisotopicMass",
            "AnnoTypeCount",
        ],
    )
    rows = []
    invalid = 0
    mismatch = 0
    mass_disagreement = 0
    for i, r in enumerate(raw.itertuples()):
        mol = Chem.MolFromSmiles(r.SMILES)
        if (
            mol is None
            or not mol.GetNumAtoms()
            or not np.isfinite(r.MonoisotopicMass)
            or r.MonoisotopicMass <= 0
        ):
            invalid += 1
            continue
        key = Chem.MolToInchiKey(mol)[:14]
        if key != r.FirstBlock:
            mismatch += 1
            continue
        if abs(Descriptors.ExactMolWt(mol) - r.MonoisotopicMass) > 0.01:
            mass_disagreement += 1
            continue
        rows.append(
            {
                "inchikey14": key,
                "normalized_smiles": Chem.MolToSmiles(mol),
                "mass": float(r.MonoisotopicMass),
                "origin": "pubchemlite",
                "pubchem_cid": int(r.Identifier),
                "annotation_count": int(r.AnnoTypeCount),
            }
        )
        if (i + 1) % 100000 == 0:
            print("validated structures", i + 1, flush=True)
    frame = pd.DataFrame(rows).drop_duplicates("inchikey14")
    frame.to_parquet(CATALOG, index=False)
    write_json(
        EXTERNAL / "manifest.json",
        {
            "doi": meta["doi"],
            "license": meta["metadata"]["license"]["id"],
            "source_sha256": sha(path),
            "derived_sha256": sha(CATALOG),
            "input_rows": len(raw),
            "structures": len(frame),
            "invalid": invalid,
            "identity_mismatch": mismatch,
            "mass_disagreement": mass_disagreement,
            "filtering": "structure validity and public source identity/mass consistency only; no query labels used",
        },
    )


def protocol():
    ROOT.mkdir(parents=True, exist_ok=True)
    spec = {
        "external_sha256": sha(CATALOG),
        "encoder_sha256": sha(ENCODER),
        "configs": [0.0, 0.25, 0.5, 1.0],
        "change": "add distinct PubChemLite structure keys to existing train+COCONUT common pool; unchanged 35ppm/floor.006 filter",
        "ranking": "same frozen 60K fingerprint encoder, .75 neural fusion; historical confidence>=.5 unchanged; blend original and expanded full rankings by RRF at weights 0/.25/.5/1",
        "development": "previous matched single-query dev2000; same known/unknown reference caches",
        "selection": "maximum unknown MRR, known MRR>=baseline-.001 and top1>=baseline-.005; strict unknown improvement; smaller weight breaks ties",
        "acceptance": "new2000 unknown paired CI95 lower>0, known MRR delta>=-.001 and top1 delta>=-.005",
        "limits": "single-query controlled validation, public candidate sources only; latest competition external-data terms not verified because browser access was declined",
    }
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == spec
    write_json(path, spec)
    return spec


def data_paths(split, mode):
    if split == "matcheddev":
        return (
            PREVIOUS / f"{split}.parquet",
            PREVIOUS / f"{split}_{mode}_records.json",
            PREVIOUS / "reference" / f"{split}_{mode}",
        )
    return (
        ROOT / f"{split}.parquet",
        ROOT / f"{split}_{mode}_records.json",
        ROOT / "reference" / f"{split}_{mode}",
    )


def records(split, mode):
    out = ROOT / f"{split}_{mode}_rankings.json"
    if out.exists():
        return json.loads(out.read_text())
    framepath, controlpath, refdir = data_paths(split, mode)
    frame = pd.read_parquet(framepath)
    assert frame.inchikey14.is_unique
    control = json.loads(controlpath.read_text())
    reference = ReferenceIndex(refdir)
    catalog = pd.read_parquet(OLD / "catalog.parquet")
    if mode == "known":
        observed = set(reference.rows.inchikey14)
        catalog = catalog[
            ((catalog.split == "train") | catalog.inchikey14.isin(frame.inchikey14))
            & catalog.inchikey14.isin(observed)
        ]
    oldpool = build_candidates(catalog, COCONUT, final=mode == "known")
    external = pd.read_parquet(CATALOG)
    index = CandidateIndex(expanded_pool(oldpool, external))
    oldkeys = set(oldpool.inchikey14)
    external_keys = set(external.inchikey14)
    probabilities = ROOT / f"encoder_{split}_probabilities.npy"
    if probabilities.exists():
        probs = np.load(probabilities)
    else:
        model, checkpoint = load_deployment_checkpoint(ENCODER, "scale")
        probs = np.stack(
            [
                group_probability(model, frame.iloc[[i]], checkpoint["preprocessing"])
                for i in range(len(frame))
            ]
        )
        np.save(probabilities, probs)
        del model
    positions = {key: i for i, key in enumerate(frame.inchikey14)}
    result = []
    for i, r in enumerate(control):
        key = r["key"]
        group = frame.iloc[[positions[key]]]
        center = center_mass(group)
        candidates, fps = index.fps(index.query(center))
        keys = candidates.inchikey14.tolist()
        mask = np.array([k in oldkeys for k in keys], dtype=bool)
        original = candidates[mask]
        oldfps = fps[mask]
        old_ids = original.inchikey14.tolist()
        assert set(old_ids) == set(r["available"]["union35"]), key
        probability = probs[positions[key]]
        old_neural = neural_rank(probability, old_ids, oldfps)
        old_rank = low_confidence_rank(
            r["rankings"]["coconut15"], r["current_full"], old_neural, 0.75
        )
        new_neural = neural_rank(probability, keys, fps)
        if r["confidence"] >= 0.5:
            old_rank = r["rankings"]["coconut15"]
            new_rank = old_rank
        else:
            current = baseline_rank(group, candidates, fps, reference, center)
            new_rank = low_confidence_rank(
                r["rankings"]["coconut15"], current, new_neural, 0.75
            )
        result.append(
            {
                "key": key,
                "known": r["known"],
                "confidence": r["confidence"],
                "old": old_rank,
                "expanded": new_rank,
                "old_pool": list(set(old_ids) | set(r["available"]["coconut15"])),
                "expanded_pool": list(set(keys) | set(r["available"]["coconut15"])),
                "old_candidates": len(old_ids),
                "expanded_candidates": len(keys),
                "in_external": key in external_keys,
                "source": group.iloc[0].ingest_lib,
            }
        )
        if (i + 1) % 250 == 0:
            print(split, mode, i + 1, flush=True)
    write_json(out, result)
    return result


def evaluate_rows(rows, weight, mode):
    rankings = {}
    pools = {}
    for r in rows:
        if mode == "known" and not r["known"]:
            continue
        key = r["key"]
        pools[key] = r["expanded_pool"] if weight else r["old_pool"]
        rankings[key] = (
            r["old"]
            if not weight or r["confidence"] >= 0.5
            else rrf([r["old"], r["expanded"]], [1 - weight, weight])
        )
    return metrics(rankings, pools)


def develop(spec):
    path = ROOT / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = {m: records("matcheddev", m) for m in ["unknown", "known"]}
    baseline = {m: evaluate_rows(rows[m], 0, m)[0] for m in rows}
    comparisons = []
    for weight in spec["configs"]:
        reports = {m: evaluate_rows(rows[m], weight, m)[0] for m in rows}
        eligible = (
            reports["unknown"]["mrr25"] >= baseline["unknown"]["mrr25"]
            and reports["known"]["mrr25"] >= baseline["known"]["mrr25"] - 0.001
            and reports["known"]["top1"] >= baseline["known"]["top1"] - 0.005
        )
        comparisons.append({"weight": weight, **reports, "eligible": eligible})
    selected = max(
        (r for r in comparisons if r["eligible"]),
        key=lambda r: (r["unknown"]["mrr25"], -r["weight"]),
    )
    selection = {
        "winner": selected,
        "baseline": baseline,
        "comparisons": comparisons,
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "frozen_before_fresh": True,
    }
    write_json(path, selection)
    diagnostic = pd.DataFrame(
        [
            {
                "key": r["key"],
                "source": r["source"],
                "old_covered": r["key"] in r["old_pool"],
                "expanded_covered": r["key"] in r["expanded_pool"],
                "in_external": r["in_external"],
                "old_candidates": r["old_candidates"],
                "expanded_candidates": r["expanded_candidates"],
            }
            for r in rows["unknown"]
        ]
    )
    diagnostic.to_csv(ROOT / "dev_coverage.csv", index=False)
    print("SELECTION", json.dumps(selection), flush=True)
    return selection


def fresh_data():
    if (ROOT / "fresh.parquet").exists():
        return
    used = set()
    paths = []
    for directory, names in [
        (
            OLD,
            ["train", "dev", "holdout", "fresh_holdout", "diagnostic", "final_train"],
        ),
        (Path("artifacts/ablation_20260928"), ["fresh"]),
        (SCALE, ["fresh", "train60k"]),
        (Path("artifacts/router_20260929"), ["fresh"]),
        (PREVIOUS, ["fresh"]),
    ]:
        for name in names:
            path = directory / f"{name}.parquet"
            paths.append(str(path))
            used.update(pd.read_parquet(path, columns=["inchikey14"]).inchikey14)
    catalog = pd.read_parquet(OLD / "catalog.parquet")
    keys = set(
        catalog.loc[
            (catalog.split == "holdout") & ~catalog.inchikey14.isin(used), "inchikey14"
        ].head(2000)
    )
    assert len(keys) == 2000 and not keys & used
    frame = single_query(sample_keys(keys, 20261004))
    frame.to_parquet(ROOT / "fresh.parquet", index=False)
    write_json(
        ROOT / "fresh_manifest.json",
        {
            "molecules": len(keys),
            "prior_overlap": 0,
            "excluded_paths": paths,
            "selection_sha256": sha(ROOT / "selection.json"),
            "sha256": sha(ROOT / "fresh.parquet"),
        },
    )


def acceptance(selection):
    path = ROOT / "fresh_report.json"
    if path.exists():
        return json.loads(path.read_text())
    if not selection["winner"]["weight"]:
        report = {
            "accepted": False,
            "reason": "No development improvement; no new holdout exposed",
        }
        write_json(path, report)
        return report
    fresh_data()
    previous = ablation.ROOT
    ablation.ROOT = ROOT
    try:
        for mode in ["unknown", "known"]:
            ablation.build_records("fresh", mode)
    finally:
        ablation.ROOT = previous
    reports = {}
    for mode in ["unknown", "known"]:
        data = records("fresh", mode)
        new, a = evaluate_rows(data, selection["winner"]["weight"], mode)
        old, b = evaluate_rows(data, 0, mode)
        a.to_csv(ROOT / f"fresh_{mode}_selected.csv", index=False)
        b.to_csv(ROOT / f"fresh_{mode}_baseline.csv", index=False)
        new.update(baseline=old, difference=bootstrap_difference(a, b))
        reports[mode] = new
        gc.collect()
        print("FRESH", mode, json.dumps(new), flush=True)
    reports["accepted"] = (
        reports["unknown"]["difference"]["ci95"][0] > 0
        and reports["known"]["difference"]["difference"] >= -0.001
        and reports["known"]["top1"] >= reports["known"]["baseline"]["top1"] - 0.005
    )
    write_json(path, reports)
    return reports


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["catalog", "develop", "run"])
    args = parser.parse_args()
    configure(42, 4)
    prepare_catalog()
    if args.stage == "catalog":
        return
    spec = protocol()
    selection = develop(spec)
    if args.stage == "run":
        print("FINAL", json.dumps(acceptance(selection)), flush=True)


if __name__ == "__main__":
    main()
