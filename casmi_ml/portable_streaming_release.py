"""Fresh runtime-only amendment retaining the frozen236 model and science recipe."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path("artifacts/research_loop/rounds/0357_streaming_prefetch_zero_control")


def prepare(old, output):
    old, output = Path(old), Path(output)
    if not Controller().local_only() or output.exists():
        raise ValueError("Fresh local-only runtime amendment required")
    control = json.loads((CONTROL / "report.json").read_text())
    if (
        not control["control_passed"]
        or control["development_budget_fallbacks"]
        or control["incremental_new_failure_contexts"]
        or control["streaming_source_sha256"]
        != digest("casmi_ml/streaming_fragment_prefetch.py")
        or control["prefetch_source_sha256"]
        != digest("casmi_ml/portable_fragment_prefetch.py")
    ):
        raise ValueError("Current full3340 exact streaming control required")
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Previous release source changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("streaming_fragment_prefetch.py", "portable_streaming_pipeline.py"):
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["streaming_fragment_prefetch"] = {
        "source_sha256": digest("casmi_ml/streaming_fragment_prefetch.py"),
        "prefetch_source_sha256": digest("casmi_ml/portable_fragment_prefetch.py"),
        "control_report_sha256": digest(CONTROL / "report.json"),
        "full_rank_matches": {"unknown": 2000, "known": 1340},
        "physical_workers": 2,
        "scope": "Runtime-only overlap of unchanged physical request discovery and execution. Same original model/science/deadline/timeout/threads2. New fresh75+cold400+platform400 fullrank/resource proof required; no query-dependent cache packaged.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    p = output / "notebook/casmi_chemistry.ipynb"
    notebook = json.loads(p.read_text())
    notebook["cells"][1]["source"] = notebook["cells"][1]["source"].replace(
        "casmi_ml.portable_prefetch_pipeline", "casmi_ml.portable_streaming_pipeline"
    )
    write_json(p, notebook)
    dataset_id = "giaok246/casmi26-236-streaming-best-bundle"
    metadata = json.loads((output / "notebook/kernel-metadata.json").read_text())
    metadata.update(
        id="giaok246/casmi26-236-streaming-verified-best",
        title="CASMI26 236 Streaming Verified Best",
        dataset_sources=[dataset_id, "aidensong123/casmi26-coconut-202609"],
    )
    write_json(output / "notebook/kernel-metadata.json", metadata)
    sums = {
        str(p.relative_to(bundle)): digest(p)
        for p in bundle.rglob("*")
        if p.is_file()
        and p.name not in ("manifest.json", "SHA256SUMS.json")
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    write_json(bundle / "SHA256SUMS.json", sums)
    (output / "dataset").mkdir()
    with zipfile.ZipFile(
        output / "dataset/casmi_chemistry_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in [*sums, "SHA256SUMS.json"]:
            archive.write(bundle / name, name)
    metadata = json.loads((old / "dataset/dataset-metadata.json").read_text())
    metadata.update(id=dataset_id, title="CASMI26 Scientific236 Streaming Assets")
    write_json(output / "dataset/dataset-metadata.json", metadata)
    write_json(
        output / "status.json",
        {
            "status": "runtime_amendment_prepared",
            "identity": release_identity(output, sums, VARIANT),
            "scientific_round": "0236_low_supported_leader_first",
            "development_eligible": True,
            "independent_acceptance": False,
            "uploaded": False,
            "submitted": False,
        },
    )
    write_json(
        output / "runtime_amendment.json",
        {
            "old_release": str(old),
            "new_release": str(output),
            "science_recipe_unchanged": True,
            "reason": "Previous platform exhausted original shared1200s deadline after729.59s frontend. Stream original physical requests during unchanged critic/probability discovery; preserve original budgets and full400 rank gate.",
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    print(prepare(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
