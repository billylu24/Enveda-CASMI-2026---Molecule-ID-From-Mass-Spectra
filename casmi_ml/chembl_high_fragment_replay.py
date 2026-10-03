"""Replay high-fragment extension on the actual frozen generator's unlabeled handoff."""

import argparse
import json
import time
from pathlib import Path

from rdkit import Chem

from casmi_ml.chembl_routed_inference import extend
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

BASE = Path("artifacts/research_loop/rounds/0149_chembl_routed_combination")


def run(directory, output):
    directory, output = Path(directory), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["prefix"] != 3 or protocol["limit"] != 2000:
        raise ValueError("Frozen prefix3/full experiment required")
    # Implementation replay can validate a rejected development experiment too;
    # packaging separately requires its actual development gate.
    rankings = json.loads((directory / "selected_rankings.json").read_text())["unknown"]
    config = json.loads((BASE / "external_config.json").read_text())
    worker = Path("artifacts/research_loop/metfrag_worker_classes/MetFragWorker.class")
    worker_source = Path("casmi_ml/java/MetFragWorker.java")
    config["high_fragment"] = {
        "prefix": 3,
        "weight": 0.5,
        "backend": "persistent",
        "worker_class": str(worker),
        "worker_class_sha256": digest(worker),
        "worker_source_sha256": digest(worker_source),
    }
    write_json(directory / "external_config.json", config)
    inputs = BASE / "baseline_replay/replay"
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "inference_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "selected_rankings_sha256": digest(directory / "selected_rankings.json"),
            "experiment_protocol_sha256": digest(directory / "protocol.json"),
            "test_sha256": digest(inputs / "test.parquet"),
            "baseline_csv_sha256": digest(inputs / "submission.csv"),
            "baseline_full_sha256": digest(inputs / "generated_full.json"),
            "external_config": config,
            "independent_acceptance": False,
            "scope": "75 original label-free queries and real frozen generator handoff; compare complete actual rankings",
        },
    )
    started = time.monotonic()
    extend(
        inputs / "test.parquet",
        inputs / "submission.csv",
        inputs / "generated_full.json",
        inputs / "routing.csv",
        output / "submission.csv",
        Path("."),
        config,
        deadline=started + 1800,
        fragment_deadline=started + 1200,
        java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
        cache="artifacts/chemistry_20261001/chembl_metfrag_cache",
        output_full_rankings=output / "full.json",
    )
    actual = json.loads((output / "full.json").read_text())
    mismatch = []
    for row in actual:
        keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in row["smiles"]]
        if keys != rankings[row["molecule_id"]]:
            mismatch.append(row["molecule_id"])
    if mismatch:
        write_json(output / "private_mismatches.json", mismatch)
        raise ValueError("Real high fragment rankings differ from frozen development")
    report = json.loads(
        Path(str(output / "submission.csv") + ".external.report.json").read_text()
    )
    result = {
        "valid": True,
        "molecules": len(actual),
        "full_rank_matches": len(actual),
        "unlabeled_input": True,
        "real_frozen_generator_baseline": True,
        "external_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
        "protocol_sha256": digest(directory / "protocol.json"),
        "seconds": report["seconds"],
        "peak_rss_mib": report["parent_peak_rss_mib"],
        "status_counts": report["status_counts"],
        "independent_acceptance": False,
        "resource_scope": "75-query implementation replay with content cache; full uncached package/platform resources still required",
    }
    write_json(directory / "replay.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.directory, a.output), indent=2))


if __name__ == "__main__":
    main()
