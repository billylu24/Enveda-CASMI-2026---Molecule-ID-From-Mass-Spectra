"""Aggregate complete query-system changes by existing observable route strata."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import evidence_features
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

ARMS = {
    "paired_raw": "0262_query_multispectrum_chemical_system",
    "chemical_only": "0268_query_chemical_only_system",
    "modality_balanced": "0274_query_balanced_chemical_system",
    "strong_peak_losses": "0283_query_strong_peak_chemical_system",
    "admission_margin": "0291_query_admission_chemical_system",
}
BASE = Path("artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay")


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Movement audit is local-only")
    output = Path(output)
    state = {r["id"]: r for r in c.read()["rounds"]}
    if any(not state.get(rid, {}).get("local_recorded") for rid in ARMS.values()):
        raise ValueError("Every diagnostic arm must have complete finalized evidence")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "baseline_rankings_sha256": digest(BASE / "selected_rankings.json"),
            "arm_rankings_sha256": {
                name: digest(c.root / "rounds" / rid / "selected_rankings.json")
                for name, rid in ARMS.items()
            },
            "scope": "Every fixed2000 unknown and1340 known molecule, complete finalized arms. Observable strata: existing route confidence/status and any precise24 chemical evidence. Truth used only to aggregate rank deltas, no data selection or proposed stratum gating.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    expected = json.loads((BASE / "selected_rankings.json").read_text())
    result = {}
    for mode in ["unknown", "known"]:
        frame = pd.read_parquet(CONTROL / mode / "test.parquet")
        chemical = {
            key: max(
                float(evidence_features(r, True).max())
                for r in group.to_dict("records")
            )
            for key, group in frame.groupby("molecule_id", sort=False)
        }
        audit = pd.read_csv(BASE / mode / "selected.csv.external.csv").set_index(
            "molecule_id"
        )
        baseline = pd.read_csv(BASE / f"{mode}_supported_leader_replay.csv").set_index(
            "key"
        )
        rows = {}
        for name, rid in ARMS.items():
            root = c.root / "rounds" / rid
            actual = json.loads((root / "selected_rankings.json").read_text())[mode]
            values = pd.read_csv(
                root / f"{mode}_chemical_query_residual.csv"
            ).set_index("key")
            strata = {}
            for key in expected[mode]:
                confidence = (
                    float(audit.loc[key, "confidence"]) if key in audit.index else None
                )
                route = (
                    "low"
                    if confidence is not None and confidence < 0.5
                    else "high"
                    if confidence is not None
                    else "empty"
                )
                evidence = "evidence" if chemical.get(key, 0) > 0 else "no_evidence"
                label = route + ":" + evidence
                entry = strata.setdefault(
                    label,
                    {
                        "molecules": 0,
                        "full_rank_changes": 0,
                        "top1_identity_changes": 0,
                        "positive_mrr_changes": 0,
                        "negative_mrr_changes": 0,
                        "total_mrr_delta": 0.0,
                    },
                )
                delta = float(
                    values.loc[key, "reciprocal_rank"]
                    - baseline.loc[key, "reciprocal_rank"]
                )
                entry["molecules"] += 1
                entry["full_rank_changes"] += int(actual[key] != expected[mode][key])
                entry["top1_identity_changes"] += int(
                    actual[key][:1] != expected[mode][key][:1]
                )
                entry["positive_mrr_changes"] += int(delta > 1e-12)
                entry["negative_mrr_changes"] += int(delta < -1e-12)
                entry["total_mrr_delta"] += delta
            rows[name] = strata
        result[mode] = rows
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "strata": result,
        "scope": "Descriptive repeated-development architecture movement and gain concentration; no stratum-based rule fitting or accuracy qualification",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
