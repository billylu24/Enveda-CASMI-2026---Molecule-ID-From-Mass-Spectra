"""Public ChEBI import and coverage diagnostic; no query-driven source selection."""

import argparse
import csv
import gzip
import json
import math
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from casmi_ml.candidate_catalog import expanded_pool
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex, build_candidates
from casmi_ml.research_protocol import CATALOG, ROOT, freeze
from casmi_ml.scale_experiment import COCONUT

RAW = Path("external/chebi")
DERIVED = RAW / "structures.parquet"


def normalize(row):
    if row.get("status_id") not in ("1", "3") or row.get("default_structure") != "true":
        return None, "not_reviewed_default"
    molecule = Chem.MolFromSmiles(row.get("smiles", ""))
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
    if len(key) != 14 or (declared and declared != key):
        return None, "identity_disagreement"
    Chem.RemoveStereochemistry(molecule)
    mass = Descriptors.ExactMolWt(molecule)
    if not math.isfinite(mass) or mass <= 0:
        return None, "invalid_mass"
    return {
        "inchikey14": key,
        "normalized_smiles": Chem.MolToSmiles(molecule, isomericSmiles=False),
        "mass": mass,
        "origin": "chebi255",
        "chebi_id": row["compound_id"],
    }, "retained"


def prepare():
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    csv.field_size_limit(20_000_000)
    if DERIVED.exists():
        return pd.read_parquet(DERIVED), json.loads((RAW / "manifest.json").read_text())
    counts, selected = Counter(), {}
    with gzip.open(RAW / "structures.tsv.gz", "rt") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            record, reason = normalize(row)
            counts[reason] += 1
            if record is not None:
                selected.setdefault(record["inchikey14"], record)
            if sum(counts.values()) % 50000 == 0:
                print("chebi_import", sum(counts.values()), len(selected), flush=True)
    frame = (
        pd.DataFrame(selected.values())
        .sort_values(["mass", "inchikey14"])
        .reset_index(drop=True)
    )
    frame.to_parquet(DERIVED, index=False)
    manifest = {
        "source": "https://www.ebi.ac.uk/chebi",
        "release": 255,
        "source_sha256": digest(RAW / "structures.tsv.gz"),
        "license_sha256": digest(RAW / "LICENSE"),
        "readme_sha256": digest(RAW / "README"),
        "derived_sha256": digest(DERIVED),
        "input_rows": sum(counts.values()),
        "structures": len(frame),
        "filters": dict(counts),
        "selection": "All reviewed default organic neutral nonisotopic specific structures; no query/truth selection",
        "normalization": "Remove stereochemistry only; preserve connectivity and intrinsic charge; no salt splitting or tautomer changes",
        "license": "CC-BY4.0 according to supplied LICENSE; retain original README which also mentions ShareAlike",
    }
    write_json(RAW / "manifest.json", manifest)
    (RAW / "ATTRIBUTION.md").write_text(
        "# ChEBI structure data\n\nChEBI data is from https://www.ebi.ac.uk/chebi — release 255, 2026-09-09.\nFull public structures.tsv.gz downloaded from https://ftp.ebi.ac.uk/pub/databases/chebi/flat_files/.\nThe supplied LICENSE states Creative Commons Attribution 4.0; retain LICENSE and README with any redistribution. The README also mentions ShareAlike.\n\nDerived data filters reviewed default, valid organic neutral structures, excludes mixtures, generic atoms and isotopes, verifies declared connectivity identity, removes stereochemistry and deduplicates connectivity keys. No query labels used in source selection. ChEBI does not endorse these predictions.\n"
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
            "raw_sha256": digest(RAW / "structures.tsv.gz"),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "existing_catalog_sha256": digest(CATALOG),
            "coconut_sha256": digest(COCONUT),
            "pubchemlite_sha256": digest("external/pubchemlite/structures.parquet"),
            "scope": "Full public import and descriptive coverage only; no ranking/submission eligibility",
            "truth_used_only_in_metrics": True,
            "holdout_used": False,
        },
    )
    derived, manifest = prepare()
    original = expanded_pool(
        build_candidates(pd.read_parquet(CATALOG), COCONUT, final=False),
        pd.read_parquet("external/pubchemlite/structures.parquet"),
    )
    expanded = expanded_pool(original, derived)
    base_index, new_index = CandidateIndex(original), CandidateIndex(expanded)
    frame = pd.read_parquet(
        ROOT / "researchdev.parquet", columns=["inchikey14", "adduct", "precursor_mz"]
    )
    keys = set(frame.inchikey14)
    base_keys, new_keys = set(original.inchikey14), set(expanded.inchikey14)
    mass_before, mass_after = 0, 0
    for key, group in frame.groupby("inchikey14", sort=True):
        # Query truth only measures coverage, never changes imported structures or masses.
        # Avoid fingerprint generation for a coverage-only audit.
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
        "scope": "Repeated2000 development coverage diagnosis only; ranking required before submission",
    }
    write_json(output / "report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2))


if __name__ == "__main__":
    main()
