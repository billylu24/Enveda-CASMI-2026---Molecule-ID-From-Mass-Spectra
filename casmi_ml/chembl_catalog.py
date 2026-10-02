"""Full public ChEMBL37 import and development coverage audit; no query selection."""

import argparse
import csv
import gzip
import json
import math
import multiprocessing
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from casmi_ml.candidate_catalog import expanded_pool
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex, build_candidates
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import CATALOG, ROOT, freeze
from casmi_ml.scale_experiment import COCONUT

RAW = Path("external/chembl37")
DERIVED = RAW / "structures.parquet"
SOURCE = RAW / "chembl_37_chemreps.txt.gz"
EXPECTED_SHA256 = "ea6181ce8dc7af41974e35b92e1febb0c9dcbe2c62f7ccc4a5d983ac19f696e7"


def normalize(row):
    molecule = Chem.MolFromSmiles(row.get("canonical_smiles", ""))
    if molecule is None or not molecule.GetNumAtoms():
        return None, "invalid"
    if len(Chem.GetMolFrags(molecule)) != 1 or not any(
        a.GetAtomicNum() == 6 for a in molecule.GetAtoms()
    ):
        return None, "mixture_or_nonorganic"
    if Chem.GetFormalCharge(molecule) or any(
        a.GetAtomicNum() == 0 or a.GetIsotope() for a in molecule.GetAtoms()
    ):
        return None, "charged_generic_or_isotopic"
    key = Chem.MolToInchiKey(molecule)[:14]
    declared = row.get("standard_inchi_key", "")[:14]
    if len(key) != 14 or len(declared) != 14 or key != declared:
        return None, "identity_disagreement"
    Chem.RemoveStereochemistry(molecule)
    mass = Descriptors.ExactMolWt(molecule)
    if not math.isfinite(mass) or mass <= 0:
        return None, "invalid_mass"
    return {
        "inchikey14": key,
        "normalized_smiles": Chem.MolToSmiles(molecule, isomericSmiles=False),
        "mass": mass,
        "origin": "chembl37",
        "chembl_id": row["chembl_id"],
    }, "retained"


def normalize_batch(rows):
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    return [normalize(row) for row in rows]


def raw_batches():
    csv.field_size_limit(20_000_000)
    with gzip.open(SOURCE, "rt") as stream:
        batch = []
        for row in csv.DictReader(stream, delimiter="\t"):
            batch.append(row)
            if len(batch) == 1000:
                yield batch
                batch = []
        if batch:
            yield batch


def prepare(deadline):
    if digest(SOURCE) != EXPECTED_SHA256:
        raise ValueError("Public source checksum mismatch")
    if DERIVED.exists():
        manifest = json.loads((RAW / "manifest.json").read_text())
        if (
            manifest["derived_sha256"] != digest(DERIVED)
            or manifest["source_sha256"] != EXPECTED_SHA256
        ):
            raise ValueError("Derived source checksum mismatch")
        return pd.read_parquet(DERIVED), manifest
    counts, selected = Counter(), {}
    with multiprocessing.get_context("spawn").Pool(8) as pool:
        for results in pool.imap(normalize_batch, raw_batches(), chunksize=1):
            if time.monotonic() >= deadline:
                raise TimeoutError("Catalog normalization exceeded CPU budget")
            for record, reason in results:
                counts[reason] += 1
                if record is not None:
                    selected.setdefault(record["inchikey14"], record)
            if sum(counts.values()) % 100000 == 0:
                print(
                    "chembl_imported",
                    sum(counts.values()),
                    "unique",
                    len(selected),
                    flush=True,
                )
    frame = (
        pd.DataFrame(selected.values())
        .sort_values(["mass", "inchikey14"])
        .reset_index(drop=True)
    )
    frame.to_parquet(DERIVED, index=False)
    manifest = {
        "source": "https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_37/",
        "release": 37,
        "release_date": "2026-05-01",
        "source_sha256": EXPECTED_SHA256,
        "license_sha256": digest(RAW / "LICENSE"),
        "attribution_sha256": digest(RAW / "REQUIRED.ATTRIBUTION"),
        "derived_sha256": digest(DERIVED),
        "input_rows": sum(counts.values()),
        "structures": len(frame),
        "filters": dict(counts),
        "selection": "All valid neutral organic connected nonisotopic specific public structures; no query/truth selection",
        "normalization": "remove stereochemistry only; preserve connectivity; validate declared InChI connectivity key; no salt splitting or tautomer changes",
        "license": "CC-BY-SA3.0, retain original license and attribution with derived data",
    }
    write_json(RAW / "manifest.json", manifest)
    (RAW / "ATTRIBUTION.md").write_text(
        "# ChEMBL37 structures\n\nChEMBL data is from https://www.ebi.ac.uk/chembl — release37, May2026. Source chemreps file downloaded in full from the EBI public FTP HTTPS endpoint. Original LICENSE, README, REQUIRED.ATTRIBUTION and checksums retained. Data and derived structures are CC Attribution-ShareAlike3.0. ChEMBL does not endorse these predictions. No query labels used in source selection or normalization.\n"
    )
    return frame, manifest


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "raw_sha256": EXPECTED_SHA256,
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "existing_catalog_sha256": digest(CATALOG),
            "coconut_sha256": digest(COCONUT),
            "pubchemlite_sha256": digest("external/pubchemlite/structures.parquet"),
            "scope": "Full public source coverage audit only; no ranking/submission eligibility",
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
            "cohort": "repeated_development",
            "cpu_seconds": 3600,
            "workers": 8,
        },
    )
    budget = StageBudget(
        output,
        "catalog_cpu",
        "chembl37",
        3600,
        limit=3600,
        lock_path=output / "catalog.lock",
    )
    try:
        derived, manifest = prepare(budget.started + budget.allowance)
        original = expanded_pool(
            build_candidates(pd.read_parquet(CATALOG), COCONUT, final=False),
            pd.read_parquet("external/pubchemlite/structures.parquet"),
        )
        expanded = expanded_pool(original, derived)
        base_index, new_index = CandidateIndex(original), CandidateIndex(expanded)
        frame = pd.read_parquet(
            ROOT / "researchdev.parquet",
            columns=["inchikey14", "adduct", "precursor_mz"],
        )
        keys = set(frame.inchikey14)
        base_keys, new_keys = set(original.inchikey14), set(expanded.inchikey14)
        mass_before, mass_after = 0, 0
        for key, group in frame.groupby("inchikey14", sort=True):
            from casmi_ml.mass_candidates import mass_centers

            before, after = set(), set()
            for center in mass_centers(group, "charge_aware_union"):
                before.update(base_index.query(center).inchikey14)
                after.update(new_index.query(center).inchikey14)
            mass_before += int(key in before)
            mass_after += int(key in after)
        result = {
            "diagnostic_only": True,
            "independent_acceptance": False,
            "molecules": len(keys),
            "manifest": manifest,
            "existing_catalog_structures": len(original),
            "combined_catalog_structures": len(expanded),
            "new_public_structures": len(expanded) - len(original),
            "database_truth_coverage_before": len(keys & base_keys),
            "database_truth_coverage_after": len(keys & new_keys),
            "mass_window_truth_coverage_before": mass_before,
            "mass_window_truth_coverage_after": mass_after,
            "scope": "Repeated2000 coverage diagnosis only; no combined ranking or accuracy claim",
            "seconds": time.monotonic() - budget.started,
        }
        write_json(output / "report.json", result)
        return result
    finally:
        budget.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))


if __name__ == "__main__":
    main()
