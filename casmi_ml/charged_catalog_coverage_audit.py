"""Audit the complete existing public source's omitted charged organic structures."""

import argparse
import csv
import gzip
import json
import math
import re
import time
from collections import Counter
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors

from casmi_ml.chembl_catalog import DERIVED, EXPECTED_SHA256, SOURCE
from casmi_ml.data import write_json
from casmi_ml.mass_candidates import mass_centers
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, freeze

CHARGE_TOKEN = re.compile(r"\[[^\]]*[+-][^\]]*\]")


def normalize_charged(row):
    molecule = Chem.MolFromSmiles(row.get("canonical_smiles", ""))
    if molecule is None or not molecule.GetNumAtoms():
        return None, "invalid"
    charge = Chem.GetFormalCharge(molecule)
    if not charge:
        return None, "net_neutral"
    if not 1 <= abs(charge) <= 8:
        return None, "charge_out_of_supported_range"
    if len(Chem.GetMolFrags(molecule)) != 1 or not any(
        a.GetAtomicNum() == 6 for a in molecule.GetAtoms()
    ):
        return None, "mixture_or_nonorganic"
    if any(a.GetAtomicNum() == 0 or a.GetIsotope() for a in molecule.GetAtoms()):
        return None, "generic_or_isotopic"
    key = Chem.MolToInchiKey(molecule)[:14]
    if len(key) != 14 or key != row.get("standard_inchi_key", "")[:14]:
        return None, "identity_disagreement"
    mass = Descriptors.ExactMolWt(molecule)
    if not math.isfinite(mass) or mass <= 0:
        return None, "invalid_mass"
    Chem.RemoveStereochemistry(molecule)
    return {
        "inchikey14": key,
        "normalized_smiles": Chem.MolToSmiles(molecule, isomericSmiles=False),
        "mass": mass,
        "formal_charge": charge,
        "origin": "chembl37_charged",
        "chembl_id": row["chembl_id"],
    }, "retained"


def run(output):
    output = Path(output)
    if digest(SOURCE) != EXPECTED_SHA256:
        raise ValueError("Complete public source hash differs")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "raw_source_sha256": EXPECTED_SHA256,
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "neutral_catalog_sha256": digest(DERIVED),
            "scope": "Scan full existing public chemreps source, all charged atom tokens then actual net-charge1-8, connected organic specific nonisotopic structures with declared connectivity key exact. No truth-based filtering, neutralization, salt splitting or candidate activation.",
            "candidate_mass": "Actual charged RDKit ExactMolWt retained unchanged; compare existing charge-aware union centers using35ppm/floor.006Da only for diagnostic upper bound.",
            "local_only": True,
            "independent_acceptance": False,
            "holdout_used": False,
            "truth_used_only_in_metrics": True,
        },
    )
    RDLogger.DisableLog("rdApp.warning")
    RDLogger.DisableLog("rdApp.error")
    started = time.monotonic()
    counts = Counter()
    records = {}
    csv.field_size_limit(20_000_000)
    with gzip.open(SOURCE, "rt") as stream:
        for row in csv.DictReader(stream, delimiter="\t"):
            counts["source_rows"] += 1
            if not CHARGE_TOKEN.search(row.get("canonical_smiles", "")):
                continue
            record, reason = normalize_charged(row)
            counts[reason] += 1
            if record is not None:
                records.setdefault(record["inchikey14"], record)
    catalog = pd.DataFrame(list(records.values()))
    if catalog.empty:
        raise ValueError("No supported charged structures in complete source")
    catalog.sort_values(["mass", "inchikey14"]).reset_index(drop=True).to_parquet(
        output / "charged_structures.parquet", index=False
    )
    neutral = set(pd.read_parquet(DERIVED, columns=["inchikey14"]).inchikey14)
    dev = pd.read_parquet(ROOT / "researchdev.parquet")
    masses = {r.inchikey14: r.mass for r in catalog.itertuples()}
    database_hits = window_hits = extra_hits = 0
    charges = Counter()
    for key, group in dev.groupby("inchikey14", sort=True):
        if key not in masses:
            continue
        database_hits += 1
        actual = mass_centers(group, "charge_aware_union")
        inside = any(
            abs(masses[key] - center) <= max(center * 35e-6, 0.006) for center in actual
        )
        window_hits += int(inside)
        extra_hits += int(inside and key not in neutral)
        charges[
            str(int(catalog.set_index("inchikey14").loc[key, "formal_charge"]))
        ] += 1
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "counts": dict(counts),
        "retained_unique_structures": len(catalog),
        "charged_database_truth_upper_bound": database_hits,
        "charged_observable_mass_window_truth_upper_bound": window_hits,
        "new_over_neutral_chembl_truth_upper_bound": extra_hits,
        "hit_charge_counts": dict(charges),
        "seconds": time.monotonic() - started,
        "source_catalog_sha256": digest(output / "charged_structures.parquet"),
        "activated": False,
        "scope": "Repeated2000 coverage diagnosis only; does not establish candidate addition beyond current union, correct ionic fragmentation, ranking improvement or known protection",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
