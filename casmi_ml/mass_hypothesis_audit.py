"""Coverage-only comparison of legacy and charge/multiplicity-aware mass hypotheses."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.chemistry import neutral_mass
from casmi_ml.data import write_json
from casmi_ml.ranking import CandidateIndex, build_candidates, center_mass
from casmi_ml.research_protocol import CATALOG, ROOT
from casmi_ml.scale_experiment import COCONUT


def audit(root=ROOT, split="researchdev"):
    root = Path(root)
    frame = pd.read_parquet(root / f"{split}.parquet")
    pool = build_candidates(pd.read_parquet(CATALOG), COCONUT)
    index = CandidateIndex(pool)
    universe = set(pool.inchikey14)
    output = []
    for key, group in frame.groupby("inchikey14", sort=True):
        masses = [
            m
            for row in group.to_dict("records")
            if (m := neutral_mass(row)) is not None
        ]
        centers = {
            "legacy": [center_mass(group)],
            "charge_aware_median": [float(np.median(masses))] if masses else [],
            "charge_aware_union": masses,
        }
        row = {
            "key": key,
            "source": "|".join(sorted(set(group.ingest_lib))),
            "adducts": "|".join(sorted(set(group.adduct))),
            "in_database": key in universe,
        }
        for variant, values in centers.items():
            candidates = set()
            for mass in values:
                candidates.update(index.query(mass).inchikey14)
            row[variant + "_hit"] = key in candidates
            row[variant + "_candidates"] = len(candidates)
        output.append(row)
    table = pd.DataFrame(output)
    table.to_csv(root / f"{split}_mass_hypotheses.csv", index=False)
    report = {
        "molecules": len(table),
        "in_database": int(table.in_database.sum()),
        "variants": {
            name: {
                "hits": int(table[name + "_hit"].sum()),
                "recall": float(table[name + "_hit"].mean()),
                "median_candidates": float(table[name + "_candidates"].median()),
            }
            for name in ["legacy", "charge_aware_median", "charge_aware_union"]
        },
        "coverage_only": True,
        "ranking_unchanged": True,
        "no_labels_used_to_generate_hypotheses": True,
    }
    write_json(root / f"{split}_mass_hypotheses.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    a = p.parse_args()
    print(json.dumps(audit(a.root), indent=2))


if __name__ == "__main__":
    main()
