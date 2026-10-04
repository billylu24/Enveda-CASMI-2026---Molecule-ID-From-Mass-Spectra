"""Freeze a fresh CPU-worker runtime amendment, retaining all original science."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.overlap_cpu_scoring_control import COLD, EVENTS, check_controls
from casmi_ml.portable_verified_overlap_cpu_pipeline import SOURCES
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path(
    "artifacts/research_loop/rounds/0412_overlap_cpu_scoring_full3340_control"
)


def prepare(old, output):
    old, output = Path(old), Path(output)
    if not Controller().local_only() or output.exists():
        raise ValueError("Fresh local-only runtime amendment required")
    check_controls()
    report = json.loads((CONTROL / "report.json").read_text())
    if (
        not report["control_passed"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete exact3340 scoring control required")
    bindings = json.loads((CONTROL / "cpu_control_sources.json").read_text())
    for name, sha in bindings["sources"].items():
        if digest("casmi_ml/" + name + ".py") != sha:
            raise ValueError("CPU scoring control source changed: " + name)
    if bindings["control_source_sha256"] != digest(
        "casmi_ml/overlap_cpu_scoring_control.py"
    ):
        raise ValueError("CPU scoring control changed")
    overlap = json.loads((CONTROL / "overlap_control_sources.json").read_text())
    for field, name in {
        "control_source_sha256": "overlap_scoring_full_control",
        "overlap_source_sha256": "generation_fragment_overlap",
        "stream_source_sha256": "generation_handoff_stream",
        "pipeline_source_sha256": "portable_overlap_pipeline",
    }.items():
        if overlap[field] != digest("casmi_ml/" + name + ".py"):
            raise ValueError("Original overlap scoring source changed: " + name)
    actual = json.loads((CONTROL / "selected_rankings.json").read_text())
    expected = json.loads(
        Path(
            "artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay/selected_rankings.json"
        ).read_text()
    )
    if (
        actual != expected
        or len(actual["unknown"]) != 2000
        or len(actual["known"]) != 1340
    ):
        raise ValueError("Complete original3340 rankings differ")
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Original immutable bundle changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in SOURCES:
        shutil.copy2("casmi_ml/" + name + ".py", bundle / "casmi_ml" / (name + ".py"))
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["generation_cpu_runtime"] = {
        "sources": {name: digest("casmi_ml/" + name + ".py") for name in SOURCES},
        "torch_cpu_threads": 1,
        "full_rank_matches": {"unknown": 2000, "known": 1340},
        "all400_generation_events_exact": True,
        "all400_generation_csv_audit_fullhandoffs_exact": True,
        "all400_top25_exact": True,
        "all265_raw_physical_dictionaries_exact": True,
        "generation_control_sha256": digest(EVENTS / "report.json"),
        "cold_control_sha256": digest(COLD / "report.json"),
        "scoring_control_sha256": digest(CONTROL / "report.json"),
        "cpu_control_sources_sha256": digest(CONTROL / "cpu_control_sources.json"),
        "scope": "Original scientific236 models/recipe/seed/128samples/scores/critic/candidates and twoJVM/two candidate threads unchanged. Only original generator Torch CPU threads4->1. No query-dependent assets packaged. Original60s request/1200s physical/1800s total/8192MiB parentRSS retained. Separate frozen75/newcold400/platform400 proof required before the same unused one-time POST; no accuracy or paired speed claim.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    notebook = output / "notebook/casmi_chemistry.ipynb"
    document = json.loads(notebook.read_text())
    document["cells"][1]["source"] = document["cells"][1]["source"].replace(
        "casmi_ml.portable_verified_overlap_pipeline",
        "casmi_ml.portable_verified_overlap_cpu_pipeline",
    )
    write_json(notebook, document)
    dataset = "giaok246/casmi26-236-overlap-cpu-best-bundle"
    metadata = json.loads((output / "notebook/kernel-metadata.json").read_text())
    metadata.update(
        id="giaok246/casmi26-236-overlap-cpu-verified-best",
        title="CASMI26 236 Overlap CPU Verified Best",
        dataset_sources=[dataset, "aidensong123/casmi26-coconut-202609"],
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
    metadata.update(id=dataset, title="CASMI26 Scientific236 Overlap CPU Assets")
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
            "reason": "Only original seeded generator CPU4->1 after strict400 events/cold400/rawdict/full3340 controls. No change to physics/models/recipe/deadlines. Same unused explicit one-time submission request.",
        },
    )
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True)
    parser.add_argument("--output", required=True)
    prepare(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
