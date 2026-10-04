"""Prepare immutable same-science236 CPU scheduling amendment after full3340 proof."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path("artifacts/research_loop/rounds/0385_balanced_cpu_fragment_full_control")


def prepare(old, output):
    old, output = Path(old), Path(output)
    if not Controller().local_only() or output.exists():
        raise ValueError("Fresh local-only runtime amendment required")
    report = json.loads((CONTROL / "report.json").read_text())
    sources = json.loads((CONTROL / "balanced_cpu_sources.json").read_text())
    if (
        not report["control_passed"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete physical exact control required")
    if report["balanced_cpu_source_sha256"] != digest(
        "casmi_ml/balanced_cpu_fragment_runtime.py"
    ) or sources["source_sha256"] != digest(
        "casmi_ml/balanced_cpu_fragment_control.py"
    ):
        raise ValueError("Source-bound current CPU runtime/control required")
    if any(
        report[mode]["full_rank_matches"] != count
        for mode, count in [("unknown", 2000), ("known", 1340)]
    ):
        raise ValueError("All3340 original rankings must be exact")
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Previous immutable scientific bundle changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in (
        "balanced_cpu_fragment_runtime.py",
        "portable_balanced_cpu_pipeline.py",
    ):
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["balanced_cpu_runtime"] = {
        "runtime_source_sha256": sources["runtime_source_sha256"],
        "control_source_sha256": sources["source_sha256"],
        "control_report_sha256": digest(CONTROL / "report.json"),
        "full3340_rank_identity": True,
        "full_rank_matches": {"unknown": 2000, "known": 1340},
        "scope": "Only external torch scoring concurrency4->1, same-invocation complete catalog/index reuse; same original seeds/scoring graph/models/candidates/order, two JVMs each2candidate threads and shared deadlines. Fresh source-bound75/cold400 raw dictionary/resource/rank proof and platform400 still required. No query evidence packaged.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    notebook = output / "notebook/casmi_chemistry.ipynb"
    n = json.loads(notebook.read_text())
    n["cells"][1]["source"] = n["cells"][1]["source"].replace(
        "casmi_ml.portable_parallel_generation_pipeline",
        "casmi_ml.portable_balanced_cpu_pipeline",
    )
    write_json(notebook, n)
    dataset = "giaok246/casmi26-236-cpubal-best-bundle"
    metadata = output / "notebook/kernel-metadata.json"
    m = json.loads(metadata.read_text())
    m.update(
        id="giaok246/casmi26-236-cpubal-verified-best",
        title="CASMI26 236 CPU Balanced Verified Best",
        dataset_sources=[dataset, "aidensong123/casmi26-coconut-202609"],
    )
    write_json(metadata, m)
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
    m = json.loads((old / "dataset/dataset-metadata.json").read_text())
    m.update(id=dataset, title="CASMI26 Scientific236 CPU Balanced Assets")
    write_json(output / "dataset/dataset-metadata.json", m)
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
            "reason": "DualT4 generation217s and retrieval63s still exhaust original1200s physical deadline. Reduce external CPU scoring competition with original JVMs and avoid second whole-catalog construction; same models/recipe and all resource/physical/ranking gates. Same unused single submission authorization.",
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
