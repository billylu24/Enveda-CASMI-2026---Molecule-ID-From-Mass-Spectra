"""Prepare isolated unchanged236 overlap assets after complete source-bound controls."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.portable_verified_overlap_pipeline import SOURCES
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path(
    "artifacts/research_loop/rounds/0402_overlap_scoring_batched_full3340_control"
)
EVENTS = Path(
    "artifacts/research_loop/rounds/0397_generation_completed_handoff_stream_control"
)
COLD = Path(
    "artifacts/research_loop/rounds/0400_generation_physical_overlap_corrected_cold400_control"
)


def prepare(old, output):
    old, output = Path(old), Path(output)
    if not Controller().local_only() or output.exists():
        raise ValueError("Fresh same-science local-only runtime amendment required")
    report = json.loads((CONTROL / "report.json").read_text())
    bindings = json.loads((CONTROL / "overlap_control_sources.json").read_text())
    if (
        not report["control_passed"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete original3340 physical identity control required")
    for field, file in {
        "control_source_sha256": "overlap_scoring_full_control",
        "overlap_source_sha256": "generation_fragment_overlap",
        "stream_source_sha256": "generation_handoff_stream",
        "pipeline_source_sha256": "portable_overlap_pipeline",
    }.items():
        if bindings[field] != digest("casmi_ml/" + file + ".py"):
            raise ValueError("Control source binding changed: " + file)
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
        raise ValueError("All original3340 complete rankings must remain exact")
    events = json.loads((EVENTS / "report.json").read_text())
    cold = json.loads((COLD / "report.json").read_text())
    if (
        not events["all400_completed_events_exact"]
        or not events["all_generation_csv_audit_fullhandoffs_exact"]
        or events["stream_source_sha256"]
        != digest("casmi_ml/generation_handoff_stream.py")
    ):
        raise ValueError(
            "Current400 original completed generation event identity required"
        )
    if (
        not cold["valid"]
        or not cold["all400_top25_exact"]
        or not cold["all265_raw_physical_dictionaries_exact"]
        or cold["development_budget_fallbacks"]
    ):
        raise ValueError(
            "Original cold400 physical dictionary and ranking control required"
        )
    for field, file in {
        "stream_source_sha256": "generation_handoff_stream",
        "overlap_source_sha256": "generation_fragment_overlap",
        "pipeline_source_sha256": "portable_overlap_pipeline",
    }.items():
        if cold["overlap"][field] != digest("casmi_ml/" + file + ".py"):
            raise ValueError("Cold400 source binding changed: " + file)
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Immutable original same-science package changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for file in [*SOURCES, "portable_verified_overlap_pipeline", "overlap_scoring_full_control"]:
        shutil.copy2(
            Path("casmi_ml") / (file + ".py"), bundle / "casmi_ml" / (file + ".py")
        )
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["overlap_runtime"] = {
        "sources": {name: digest("casmi_ml/" + name + ".py") for name in SOURCES},
        "full_rank_matches": {"unknown": 2000, "known": 1340},
        "control_report_sha256": digest(CONTROL / "report.json"),
        "control_source_sha256": bindings["control_source_sha256"],
        "all400_original_generation_handoffs_and_events_exact": True,
        "generation_control_report_sha256": digest(EVENTS / "report.json"),
        "original_generation_source_sha256": events[
            "original_generation_source_sha256"
        ],
        "preliminary_cold400_control_sha256": digest(COLD / "report.json"),
        "scope": "Original models/recipe/candidates/scores/per-molecule seeds/physical dictionaries/heap/twoJVM/two candidate threads/shared deadlines unchanged. Runtime-only completed handoffs overlap physical discovery with generation; exact immutable catalog/encoder/score/digest memo within invocation. No query predictions/caches packaged. Still require frozen75/newcold400/platform400 source/resource/physical/ranking proof before single authorized competition POST; no accuracy or paired speed claim.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    notebook = output / "notebook/casmi_chemistry.ipynb"
    n = json.loads(notebook.read_text())
    n["cells"][1]["source"] = n["cells"][1]["source"].replace(
        "casmi_ml.portable_balanced_cpu_pipeline",
        "casmi_ml.portable_verified_overlap_pipeline",
    )
    write_json(notebook, n)
    dataset = "giaok246/casmi26-236-overlap-best-bundle"
    m = json.loads((output / "notebook/kernel-metadata.json").read_text())
    m.update(
        id="giaok246/casmi26-236-overlap-verified-best",
        title="CASMI26 236 Overlap Verified Best",
        dataset_sources=[dataset, "aidensong123/casmi26-coconut-202609"],
    )
    write_json(output / "notebook/kernel-metadata.json", m)
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
    m.update(id=dataset, title="CASMI26 Scientific236 Overlap Assets")
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
            "reason": "Same original236 models/full400 rankings/raw dictionaries/deadlines; start original physical planning from completed generated molecule events. Separate frozen/source/platform/resource proof required, same unused one-time request.",
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    prepare(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
