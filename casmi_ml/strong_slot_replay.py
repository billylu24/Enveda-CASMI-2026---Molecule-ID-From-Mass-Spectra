"""Replay181 from the actual label-free baseline and frozen reference scope."""

import argparse
import json
import time
from pathlib import Path

from rdkit import Chem

from casmi_ml.chembl_routed_inference import extend
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_protocol import freeze

BASE = Path("artifacts/research_loop/rounds/0149_chembl_routed_combination")


def run(directory):
    directory = Path(directory)
    output = directory / "unlabeled_replay"
    output.mkdir(parents=True, exist_ok=True)
    protocol = json.loads((directory / "protocol.json").read_text())
    if protocol["limit"] != 2000:
        raise ValueError("Full2000 experiment required")
    expected = json.loads((directory / "selected_rankings.json").read_text())["unknown"]
    config = json.loads(
        Path(
            "kaggle_release_high_fragment_0163_fastpoll_v2/bundle/deployment_recipe.json"
        ).read_text()
    )["external_routed"]
    bundle = Path("kaggle_release_high_fragment_0163_fastpoll_v2/bundle")
    for name in ("catalog", "encoder", "critic", "prior", "jar"):
        config[name] = str(bundle / config[name])
    high = config["high_fragment"]
    high["worker_class"] = str(bundle / high["worker_class"])
    high["runtime"]["process_class"] = str(bundle / high["runtime"]["process_class"])
    high["strong_slots"] = {
        "reference_threshold": 0.5,
        "original_prefix": 3,
        "slots": 3,
    }
    write_json(directory / "external_config.json", config)
    reference_path = Path(
        "artifacts/research_loop/rounds/0001_mass_v2/reference/unknown"
    )
    inputs = BASE / "baseline_replay/replay"
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(Path(__file__)),
            "inference_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "strong_slot_source_sha256": digest("casmi_ml/strong_slot_inference.py"),
            "selected_rankings_sha256": digest(directory / "selected_rankings.json"),
            "experiment_protocol_sha256": digest(directory / "protocol.json"),
            "reference_sha256": {
                name: digest(reference_path / name)
                for name in ("rows.parquet", "spectra.npz")
            },
            "unlabeled_inputs_sha256": {
                name: digest(inputs / name)
                for name in (
                    "test.parquet",
                    "submission.csv",
                    "generated_full.json",
                    "routing.csv",
                )
            },
            "external_config": config,
            "independent_acceptance": False,
            "scope": "Actual75 label-free frozen generator baseline with same observable proxy library scope as full development. Cold actual400 pipeline builds reference from provided train.parquet separately.",
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
        reference=ReferenceIndex(reference_path),
    )
    actual = json.loads((output / "full.json").read_text())
    mismatch = []
    for row in actual:
        keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in row["smiles"]]
        if keys != expected[row["molecule_id"]]:
            mismatch.append(row["molecule_id"])
    if len(actual) != 75 or mismatch:
        write_json(output / "private_mismatches.json", mismatch)
        raise ValueError("Strong-slot75 complete rankings differ from development")
    report = json.loads(
        Path(str(output / "submission.csv") + ".external.report.json").read_text()
    )
    result = {
        "valid": True,
        "molecules": 75,
        "full_rank_matches": 75,
        "unlabeled_input": True,
        "real_frozen_generator_baseline": True,
        "external_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
        "protocol_sha256": digest(directory / "protocol.json"),
        "seconds": report["seconds"],
        "peak_rss_mib": report["parent_peak_rss_mib"],
        "status_counts": report["status_counts"],
        "independent_acceptance": False,
        "resource_scope": "75 real implementation replay with existing100 evidence cache; actual400 cold reference handoff/runtime/platform still required.",
    }
    write_json(directory / "replay.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.directory), indent=2))


if __name__ == "__main__":
    main()
