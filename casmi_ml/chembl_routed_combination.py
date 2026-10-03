"""Compose frozen external-candidate arms on disjoint retrieval-confidence routes."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, freeze

LOW = Path("artifacts/research_loop/rounds/0123_chembl_merged_peak_full")
HIGH = Path("artifacts/research_loop/rounds/0148_chembl_high_confidence_tail")


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    for source in (LOW, HIGH):
        p = json.loads((source / "protocol.json").read_text())
        if p["incumbent_report_sha256"] != digest(incumbent / "report.json"):
            raise ValueError("Both routing arms must share the same frozen incumbent")
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "low_source": str(LOW),
            "high_source": str(HIGH),
            "low_protocol_sha256": digest(LOW / "protocol.json"),
            "high_protocol_sha256": digest(HIGH / "protocol.json"),
            "low_report_sha256": digest(LOW / "report.json"),
            "high_report_sha256": digest(HIGH / "report.json"),
            "threshold": 0.5,
            "rule": "Confidence<.5 exact0123 merged relative-fragment prefix3; confidence>=.5 exact0148 corrected preselect critic-margin tail prefix10; no molecule receives both modifications",
            "variants": ["baseline", "low_fragment_high_tail"],
            "new_training": False,
            "new_sampling": False,
            "cohort": "repeated_development",
            "holdout_used": False,
            "independent_acceptance": False,
            "resource_note": "Composition of two complete dev ranking arms; a single uncached unlabeled combined inference must pass release resources",
        },
    )
    report = {}
    for mode in ("unknown", "known"):
        conf = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        original = (
            pd.read_csv(incumbent / f"{mode}_confidence0.2_margin0.05.csv")
            .set_index("key")
            .sort_index()
        )
        pieces = []
        for source, variant, is_low in (
            (LOW, "fragment05_prefix3", True),
            (HIGH, "margin005_prefix10", False),
        ):
            baseline = (
                pd.read_csv(source / f"{mode}_baseline.csv")
                .set_index("key")
                .sort_index()
            )
            selected = (
                pd.read_csv(source / f"{mode}_{variant}.csv")
                .set_index("key")
                .sort_index()
            )
            if (
                not selected.index.equals(original.index)
                or not baseline.index.equals(original.index)
                or not np.allclose(
                    baseline.to_numpy(), original.to_numpy(), rtol=0, atol=1e-12
                )
            ):
                raise ValueError("Routing arms baseline/key mismatch")
            mask = np.asarray([(conf[k] < 0.5) == is_low for k in selected.index])
            inactive = selected.loc[~mask]
            if not np.allclose(
                inactive.to_numpy(),
                original.loc[inactive.index].to_numpy(),
                rtol=0,
                atol=1e-12,
            ):
                raise ValueError(
                    "Routing arm modifies queries outside its frozen scope"
                )
            pieces.append(selected.loc[mask])
        combined = pd.concat(pieces).sort_index()
        if combined.index.has_duplicates or not combined.index.equals(original.index):
            raise ValueError("Confidence routes overlap or omit queries")
        report[mode] = {}
        for name, frame in (
            ("baseline", original),
            ("low_fragment_high_tail", combined),
        ):
            report[mode][name] = {
                "molecules": len(frame),
                "candidate_recall": float(frame.covered.mean()),
                "mrr25": float(frame.reciprocal_rank.mean()),
                **{k: float(frame[k].mean()) for k in ("top1", "top5", "top25")},
                "conditional_mrr25": float(
                    frame.loc[frame.covered == 1, "reciprocal_rank"].mean()
                ),
            }
            frame.reset_index().to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostic_only"] = False
    report["independent_acceptance"] = False
    report["deployment_required"] = True
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--incumbent", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.incumbent), indent=2))


if __name__ == "__main__":
    main()
