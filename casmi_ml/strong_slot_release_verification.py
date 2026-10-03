"""Fresh actual75 label-free replay of packaged181 evidence rules."""

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


def replay(release):
    release = Path(release)
    output = release / "implementation_replay"
    if output.exists():
        raise ValueError("Fresh runtime implementation replay required")
    output.mkdir()
    bundle = release / "bundle"
    config = json.loads((bundle / "deployment_recipe.json").read_text())[
        "external_routed"
    ]
    inputs = Path(
        "artifacts/research_loop/rounds/0149_chembl_routed_combination/baseline_replay/replay"
    )
    expected_path = Path(
        "artifacts/research_loop/rounds/0181_high_strong_slot_full/selected_rankings.json"
    )
    expected = json.loads(expected_path.read_text())["unknown"]
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(Path(__file__)),
            "inference_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "bundle_inference_sha256": digest(
                bundle / "casmi_ml/chembl_routed_inference.py"
            ),
            "bundle_sums_sha256": digest(bundle / "SHA256SUMS.json"),
            "inputs_sha256": {
                name: digest(inputs / name)
                for name in (
                    "test.parquet",
                    "submission.csv",
                    "generated_full.json",
                    "routing.csv",
                )
            },
            "expected_rankings_sha256": digest(expected_path),
            "external_config": config,
            "fresh_cache": True,
            "unlabeled_input": True,
            "independent_acceptance": False,
        },
    )
    if digest("casmi_ml/chembl_routed_inference.py") != digest(
        bundle / "casmi_ml/chembl_routed_inference.py"
    ):
        raise ValueError("Actual replay source differs from packaged source")
    started = time.monotonic()
    extend(
        inputs / "test.parquet",
        inputs / "submission.csv",
        inputs / "generated_full.json",
        inputs / "routing.csv",
        output / "submission.csv",
        bundle,
        config,
        deadline=started + 1800,
        fragment_deadline=started + 1200,
        java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
        output_full_rankings=output / "full.json",
        reference=ReferenceIndex(
            "artifacts/research_loop/rounds/0001_mass_v2/reference/unknown"
        ),
    )
    actual = json.loads((output / "full.json").read_text())
    mismatch = []
    for row in actual:
        keys = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in row["smiles"]]
        if keys != expected[row["molecule_id"]]:
            mismatch.append(row["molecule_id"])
    if mismatch or len(actual) != 75:
        write_json(output / "private_mismatches.json", mismatch)
        raise ValueError("Packaged full75 ranks differ from frozen181")
    report = json.loads(
        Path(str(output / "submission.csv") + ".external.report.json").read_text()
    )
    result = {
        "valid": True,
        "molecules": 75,
        "full_rank_matches": 75,
        "fresh_fragment_cache": True,
        "seconds": report["seconds"],
        "parent_peak_rss_mib": report["parent_peak_rss_mib"],
        "status_counts": report["status_counts"],
        "source_sha256": report["source_sha256"],
        "bundle_sums_sha256": digest(bundle / "SHA256SUMS.json"),
        "independent_acceptance": False,
        "scope": "Same75 actual label-free generator handoff, complete ranks unchanged; cold400 and platform400 still required.",
    }
    write_json(release / "implementation_replay.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(replay(a.release), indent=2))


if __name__ == "__main__":
    main()
