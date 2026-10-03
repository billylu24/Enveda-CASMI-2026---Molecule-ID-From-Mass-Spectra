"""Index only already completed original-control physical scores for local reuse."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def run(output, control, benchmark):
    if not Controller().local_only():
        raise ValueError(
            "Incremental physical evidence is authorized only for local development"
        )
    output, control, benchmark = Path(output), Path(control), Path(benchmark)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((control / "protocol.json").read_text())["config"]
    proposals_path = Path(
        "artifacts/research_loop/rounds/0147_chembl_all_query_proposals/corrected_only_proposals.json"
    )
    proposals = json.loads(proposals_path.read_text())
    frames = {
        mode: pd.read_parquet(control / mode / "test.parquet")
        for mode in ("unknown", "known")
    }
    full = {
        mode: {
            r["molecule_id"]: r["smiles"]
            for r in json.loads((control / mode / "baseline_full.json").read_text())
        }
        for mode in frames
    }
    if any(
        {"inchikey14", "normalized_smiles", "molecular_formula", "fingerprint"}
        & set(frame)
        for frame in frames.values()
    ):
        raise ValueError("Physical seeding must use unlabeled query input")
    catalog = pd.read_parquet(
        config["catalog"], columns=["inchikey14", "normalized_smiles"]
    )
    needed = {k for key in full["unknown"] for k in proposals[key]}
    lookup = (
        catalog[catalog.inchikey14.isin(needed)]
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    del catalog
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "incremental_adapter_sha256": digest(
                "casmi_ml/development_incremental_metfrag.py"
            ),
            "control_protocol_sha256": digest(control / "protocol.json"),
            "benchmark_sha256": digest(benchmark / "report.json"),
            "proposals_sha256": digest(proposals_path),
            "query_input_sha256": {
                mode: digest(control / mode / "test.parquet") for mode in frames
            },
            "baseline_full_sha256": {
                mode: digest(control / mode / "baseline_full.json") for mode in frames
            },
            "catalog_sha256": digest(config["catalog"]),
            "rule": "Read only existing complete threads2 physical results matching exact100 original-control candidates and actual-first SMILES; record raw per-candidate scores or absent outputs with sourceSHA under exact observable physics and runtime context. No new Java request, no query truth, no cold claim.",
            "seconds_limit": 1200,
            "local_only": True,
            "diagnostic_only": True,
        },
    )
    high = config["high_fragment"]
    engine = DevelopmentIncrementalMetFrag(
        config["jar"],
        "artifacts/chemistry_20261001/chembl_metfrag_cache",
        java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
        classes=Path(high["worker_class"]).parent,
        threads=2,
        benchmark=benchmark,
    )
    engine.require_existing = True
    started = time.monotonic()
    counts = {}
    try:
        for mode, frame in frames.items():
            groups = frame.groupby("molecule_id", sort=False).indices
            counts[mode] = {"queries": 0, "groups_with_complete_evidence": 0}
            for key, indices in groups.items():
                if time.monotonic() - started >= 1200:
                    raise TimeoutError("Local physical seeding budget exhausted")
                originals = {structure_key(s): s for s in full[mode][key]}
                first = next(iter(originals))
                candidates = {
                    k: lookup[k]
                    for k in [k for k in proposals[key] if k not in originals][:100]
                }
                candidates[first] = originals[first]
                query = frame.iloc[indices].drop(columns="molecule_id")
                scores, fallback = score_group_merged(
                    engine,
                    query.to_dict("records"),
                    candidates,
                    deadline=started + 1200,
                )
                if fallback:
                    raise TimeoutError("Physical seeding group budget exhausted")
                counts[mode]["queries"] += 1
                counts[mode]["groups_with_complete_evidence"] += bool(scores)
            print(mode, counts[mode], flush=True)
    finally:
        engine.close()
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "counts": counts,
        "seconds": time.monotonic() - started,
        "incremental_stats": engine.incremental_stats,
        "engine_status_counts": getattr(engine, "status_counts", {}),
        "new_java_requests": engine.incremental_stats["incremental_requests"],
        "scope": "Local existing evidence indexing only; no metric or cold inference improvement claim",
    }
    if report["new_java_requests"]:
        raise ValueError("Seeding unexpectedly executed new physical scoring")
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--benchmark", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
