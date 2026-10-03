"""Extend the audited low prefix rule to each independently supported inserted slot."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import fingerprint, write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.diagnostic_system_evaluation import BEST, resolve_local_incremental
from casmi_ml.direct_models import DirectRanker, score_group
from casmi_ml.low_fragment_first import CONTROL, metric_report, promote_low_leader
from casmi_ml.low_leader_prefix_evaluation import supported_over_prefix
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze
from casmi_ml.secondary_inference import load_deployment_checkpoint
from casmi_ml.training import configure

PREFIX = Path("artifacts/research_loop/rounds/0239_low_leader_original_prefix_support")


def promote_supported_slots(original, current, inserted, critic, fragments, fallback):
    supported = [
        key
        for key in inserted
        if supported_over_prefix(key, original, fragments, fallback)
        and critic[key] > critic[original[0]] + 0.05
    ]
    return supported + [key for key in current if key not in supported], supported


@torch.inference_mode()
def run(output):
    c = Controller()
    if not c.local_only() or not c.read().get("local_verified_inference"):
        raise ValueError("Verified local-only incumbent required")
    configure(42, threads=4)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((BEST / "external_config.json").read_text())
    prefix = json.loads((PREFIX / "report.json").read_text())
    if (
        prefix["development_budget_fallbacks"]
        or prefix["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete prefix pilot required")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "prefix_rule_source_sha256": digest(
                "casmi_ml/low_leader_prefix_evaluation.py"
            ),
            "control_rankings_sha256": digest(CONTROL / "selected_rankings.json"),
            "prefix_rankings_sha256": digest(PREFIX / "selected_rankings.json"),
            "encoder_sha256": config["encoder_sha256"],
            "critic_sha256": config["critic_sha256"],
            "rule": "Among the same frozen three low-route inserted candidates, independently promote only those with raw fragment score strictly greater than every actual originalfirst3 and critic score greater than actualfirst+.05. Preserve fragment insertion order among promoted candidates and every other candidate relative order. All scoring under the same original encoder, critic and physical chemistry. No candidate addition or removal, threshold optimization or training.",
            "comparison": "Same repeated full2000/1340 cohorts versus verified best and complete original control; also report difference vs239 prefix leader safety control.",
            "local_only": True,
            "independent_acceptance": False,
            "fragment_seconds_per_mode": 1200,
        },
    )
    benchmark, inherited, _ = resolve_local_incremental(str(CONTROL), None)
    engine = DevelopmentIncrementalMetFrag(
        config["jar"],
        "artifacts/chemistry_20261001/chembl_metfrag_cache",
        java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
        classes=Path(config["high_fragment"]["worker_class"]).parent,
        threads=2,
        benchmark=benchmark,
    )
    encoder, saved = load_deployment_checkpoint(config["encoder"], "scale")
    encoder.eval()
    weights = torch.load(config["critic"], map_location="cpu", weights_only=True)
    if weights["encoder_sha256"] != digest(config["encoder"]):
        raise ValueError("Original critic binding differs")
    critic = DirectRanker("fingerprint").eval()
    critic.load_state_dict(weights["state_dict"])
    selected = json.loads((CONTROL / "selected_rankings.json").read_text())
    reference = json.loads((PREFIX / "selected_rankings.json").read_text())
    report, counts = {}, {}
    try:
        for mode in ("unknown", "known"):
            source = CONTROL / mode
            audit = pd.read_csv(source / "selected.csv.external.csv").set_index(
                "molecule_id"
            )
            raw = {
                row["molecule_id"]: row["smiles"]
                for row in json.loads((source / "baseline_full.json").read_text())
            }
            structures = {
                row["molecule_id"]: row["smiles"]
                for row in json.loads((source / "full.json").read_text())
            }
            test = pd.read_parquet(source / "test.parquet")
            groups = test.groupby("molecule_id", sort=False).indices
            deadline = time.monotonic() + 1200
            budget = eligible = multiple = promoted = 0
            for key, order in selected[mode].items():
                if (
                    key not in audit.index
                    or audit.loc[key, "status"] != "low_fragment_inserted"
                ):
                    continue
                eligible += 1
                original = [structure_key(s) for s in raw[key]]
                promote_low_leader(
                    original, order
                )  # Reject any unrelated frozen insertion.
                inserted = [k for k in order if k not in set(original)]
                lookup = {structure_key(s): s for s in structures[key]}
                lookup.update({structure_key(s): s for s in raw[key]})
                query = test.iloc[groups[key]].drop(columns=["molecule_id"])
                candidates = original[:3] + inserted
                fragments, fallback = score_group_merged(
                    engine,
                    query.to_dict("records"),
                    {k: lookup[k] for k in candidates},
                    deadline=deadline,
                )
                budget += int(fallback)
                scored = [original[0]] + inserted
                smiles = [lookup[k] for k in scored]
                values = score_group(
                    critic,
                    encoder,
                    query,
                    saved["preprocessing"],
                    pd.DataFrame({"normalized_smiles": smiles}),
                    np.stack([fingerprint(s) for s in smiles]).astype(np.float32),
                )
                if not np.isfinite(values).all():
                    raise ValueError("Nonfinite critic scores")
                scores = dict(zip(scored, values))
                result, supported = promote_supported_slots(
                    original, order, inserted, scores, fragments, fallback
                )
                if len(result) != len(order) or set(result) != set(order):
                    raise ValueError("Candidate universe changed")
                # Leader-only observable reconstruction must match the completed
                # prefix pilot for every group before testing added slots.
                leader_result = (
                    ([inserted[0]] + [k for k in order if k != inserted[0]])
                    if supported_over_prefix(inserted[0], original, fragments, fallback)
                    else order
                )
                if leader_result != reference[mode][key]:
                    raise ValueError("Matched original-prefix leader control differs")
                selected[mode][key] = result
                multiple += int(len(supported) > 1)
                promoted += len(supported)
            rows = pd.read_csv(CONTROL / f"{mode}_diagnostic_residual.csv").set_index(
                "key"
            )
            for key, order in selected[mode].items():
                rank = order.index(key) + 1 if key in order[:25] else 0
                rows.loc[key, "reciprocal_rank"] = 1 / rank if rank else 0
                for n in (1, 5, 25):
                    rows.loc[key, "top" + str(n)] = int(0 < rank <= n)
            report[mode] = {"supported_low_slots": metric_report(rows)}
            rows.reset_index().to_csv(
                output / f"{mode}_supported_low_slots.csv", index=False
            )
            counts[mode] = {
                "eligible": eligible,
                "multiple_supported_groups": multiple,
                "promoted_slots": promoted,
                "budget_fallbacks": budget,
                "mrr_change_vs_prefix_leader": report[mode]["supported_low_slots"][
                    "mrr25"
                ]
                - prefix[mode]["prefix_supported_leader"]["mrr25"],
            }
            print(mode, report[mode], counts[mode], flush=True)
    finally:
        engine.close()
    failures = engine.failed_physical_requests
    write_json(output / "private_incremental_failures.json", sorted(failures))
    report.update(
        local_only=True,
        independent_acceptance=False,
        diagnostics=counts,
        development_budget_fallbacks=sum(
            x["budget_fallbacks"] for x in counts.values()
        ),
        incremental_new_failure_contexts=len(failures - inherited),
    )
    report["diagnostic_only"] = bool(
        report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    )
    control = json.loads((CONTROL / "report.json").read_text())
    report["required_comparators"] = {
        "completed_evidence_original_model": {
            "directory": str(CONTROL),
            "variant": "diagnostic_residual",
            "metrics": {
                m: control[m]["diagnostic_residual"] for m in ("unknown", "known")
            },
        }
    }
    write_json(output / "selected_rankings.json", selected)
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
