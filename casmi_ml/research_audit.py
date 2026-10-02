"""Descriptive failure decomposition, never used to construct rankings."""

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from casmi_ml.chemistry import high_resolution, neutral_mass
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.research_protocol import ROOT


def audit(root=ROOT, split="researchdev"):
    root = Path(root)
    frame = pd.read_parquet(root / f"{split}.parquet")
    rows = []
    summary = {}
    for mode in ["unknown", "known"]:
        lookup = candidate_lookup(root, split, mode)
        records = json.loads((root / f"{split}_{mode}_records.json").read_text())
        groups = frame.groupby("inchikey14", sort=True).indices
        for r in records:
            group = frame.iloc[groups[r["key"]]]
            pool = set(r["available"]["union35"]) | set(r["available"]["coconut15"])
            reason = (
                "candidate_present"
                if r["key"] in pool
                else "mass_filter_miss"
                if r["key"] in lookup
                else "absent_from_database"
            )
            supported = [
                neutral_mass(row) is not None for row in group.to_dict("records")
            ]
            rows.append(
                {
                    "key": r["key"],
                    "mode": mode,
                    "source": "|".join(sorted(set(group.ingest_lib))),
                    "query_spectra": len(group),
                    "failure": reason,
                    "known": r["known"],
                    "supported_neutral_mass_spectra": sum(supported),
                    "precise_spectra": sum(
                        high_resolution(r) for r in group.instrument_type
                    ),
                    "confidence": r["confidence"],
                    "candidate_count": len(pool),
                }
            )
        chosen = [r for r in rows if r["mode"] == mode]
        summary[mode] = {
            "counts": dict(Counter(r["failure"] for r in chosen)),
            "oracle_mrr_upper_bound": sum(
                r["failure"] == "candidate_present" for r in chosen
            )
            / len(chosen),
            "no_supported_mass_queries": sum(
                r["supported_neutral_mass_spectra"] == 0 for r in chosen
            ),
        }
    output = pd.DataFrame(rows)
    output.to_csv(root / f"{split}_failure_audit.csv", index=False)
    summary["scope"] = (
        "Descriptive proxy; original production mass conversion remains unchanged. No ranking uses these labels."
    )
    write_json(root / f"{split}_failure_audit.json", summary)
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument(
        "--split", default="researchdev", choices=["researchdev", "researchholdout"]
    )
    a = p.parse_args()
    print(json.dumps(audit(a.root, a.split), indent=2))


if __name__ == "__main__":
    main()
