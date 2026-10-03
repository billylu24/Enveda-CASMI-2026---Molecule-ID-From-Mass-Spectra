"""Single frozen-evidence experiment promoting only the supported low-route leader."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.diagnostic_system_evaluation import BASE, BEST
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

CONTROL = Path("artifacts/research_loop/rounds/0206_full181_audited_exact_control")


def promote_low_leader(original, selected):
    inserted = [k for k in selected if k not in set(original)]
    if not inserted or insert_generated(original, inserted, 3, 3) != selected:
        raise ValueError("Frozen supported low insertion does not reconstruct exactly")
    prefix = min(3, len(original))
    if selected[prefix] != inserted[0]:
        raise ValueError("Frozen low supported leader differs")
    return [selected[prefix], *selected[:prefix], *selected[prefix + 1 :]]


def metric_report(frame):
    return {
        "molecules": len(frame),
        "candidate_recall": float(frame.covered.mean()),
        "mrr25": float(frame.reciprocal_rank.mean()),
        **{k: float(frame[k].mean()) for k in ("top1", "top5", "top25")},
        "conditional_mrr25": float(
            frame.loc[frame.covered > 0, "reciprocal_rank"].mean()
        ),
    }


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "incumbent_rankings_sha256": digest(BEST / "selected_rankings.json"),
            "incumbent_report_sha256": digest(BEST / "report.json"),
            "original_baseline_sha256": digest(BASE / "low_rankings.json"),
            "completed_control_sha256": digest(CONTROL / "report.json"),
            "control_audit_sha256": {
                mode: digest(CONTROL / mode / "selected.csv.external.csv")
                for mode in ("unknown", "known")
            },
            "rule": "Only confidence<.5 complete supported low_fragment_inserted groups: existing ChEMBL leader already wins raw fragment vsactual first and frozen critic+.05. Move exactly that leader from after available original first3 tofirst; all other candidates keep relative order and same positions where possible. No other branch, new candidate, score, threshold or training change. Independently normalize original-source identity from actual original SMILES and require every eligible group complete-rank exact with control before changing it.",
            "variants": ["baseline", "low_supported_leader_first"],
            "new_training": False,
            "new_sampling": False,
            "local_only": True,
            "cohort": "repeated_development",
            "independent_acceptance": False,
        },
    )
    original = json.loads((BASE / "low_rankings.json").read_text())["baseline"]
    incumbent = json.loads((BEST / "selected_rankings.json").read_text())
    completed = json.loads((CONTROL / "selected_rankings.json").read_text())
    reports, selected, counts = {}, {}, {}
    for mode in ("unknown", "known"):
        full = {
            r["molecule_id"]: r["smiles"]
            for r in json.loads((CONTROL / mode / "baseline_full.json").read_text())
        }
        audit = pd.read_csv(CONTROL / mode / "selected.csv.external.csv").set_index(
            "molecule_id"
        )
        rows = pd.read_csv(BEST / f"{mode}_strong_slot_prefix.csv").set_index("key")
        selected[mode] = {}
        changes = 0
        for key, ranking in incumbent[mode].items():
            source = original[mode][key]
            identity = dict(zip(source, [structure_key(s) for s in full.get(key, [])]))
            baseline = [identity.get(k, k) for k in ranking]
            normalized_original = [identity.get(k, k) for k in source]
            result = baseline
            if (
                key in audit.index
                and audit.loc[key, "status"] == "low_fragment_inserted"
            ):
                if (
                    audit.loc[key, "confidence"] >= 0.5
                    or baseline != completed[mode][key]
                ):
                    raise ValueError(
                        "Frozen eligible low evidence differs from observable completed control"
                    )
                result = promote_low_leader(normalized_original, baseline)
                changes += result != baseline
            selected[mode][key] = result
        reports[mode] = {}
        for name, orders in [
            (
                "baseline",
                {
                    key: [
                        dict(
                            zip(
                                original[mode][key],
                                [structure_key(s) for s in full.get(key, [])],
                            )
                        ).get(k, k)
                        for k in order
                    ]
                    for key, order in incumbent[mode].items()
                },
            ),
            ("low_supported_leader_first", selected[mode]),
        ]:
            frame = rows.copy()
            for key, order in orders.items():
                rank = order.index(key) + 1 if key in order else 0
                frame.loc[key, "reciprocal_rank"] = 1 / rank if 1 <= rank <= 25 else 0.0
                for n in (1, 5, 25):
                    frame.loc[key, "top" + str(n)] = int(1 <= rank <= n)
            reports[mode][name] = metric_report(frame)
            frame.reset_index().to_csv(output / f"{mode}_{name}.csv", index=False)
        counts[mode] = {"leader_promoted_groups": changes}
        print(mode, reports[mode], counts[mode], flush=True)
    reports.update(local_only=True, independent_acceptance=False, diagnostics=counts)
    reports["required_comparators"] = {
        "completed_evidence_original_model": {
            "directory": str(CONTROL),
            "variant": "diagnostic_residual",
            "metrics": {
                mode: json.loads((CONTROL / "report.json").read_text())[mode][
                    "diagnostic_residual"
                ]
                for mode in ("unknown", "known")
            },
        }
    }
    write_json(output / "selected_rankings.json", selected)
    write_json(output / "report.json", reports)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
