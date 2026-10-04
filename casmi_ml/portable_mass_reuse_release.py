"""Freeze exact scoped mass reuse after complete local original236 controls."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.mass_reuse_scoring_control import CLASSES, COLD
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path("artifacts/research_loop/rounds/0425_mass_reuse_full3340_source_control")
SOURCES = ("mass_reuse_fragment_runtime", "portable_mass_reuse_pipeline")


def prepare(old, output):
    old, output = Path(old), Path(output)
    if not Controller().local_only() or output.exists():
        raise ValueError("Fresh local-only runtime amendment required")
    cold = json.loads((COLD / "report.json").read_text())
    control = json.loads((CONTROL / "report.json").read_text())
    if not (
        cold["valid"]
        and cold["all400_top25_exact"]
        and cold["all400_generation_csv_audit_fullhandoffs_exact"]
        and cold["all265_raw_physical_dictionaries_exact"]
        and cold["development_budget_fallbacks"] == 0
        and control["control_passed"]
        and control["development_budget_fallbacks"] == 0
        and control["incremental_new_failure_contexts"] == 0
    ):
        raise ValueError(
            "Complete cold400 and source-bound3340 exact controls required"
        )
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
        raise ValueError("Original complete3340 ranks differ")
    bindings = json.loads((CONTROL / "mass_reuse_sources.json").read_text())
    if (
        digest("casmi_ml/mass_reuse_scoring_control.py") != bindings["source_sha256"]
        or digest("casmi_ml/mass_reuse_fragment_runtime.py")
        != bindings["runtime_source_sha256"]
    ):
        raise ValueError("Verified mass-reuse controls changed")
    classes = {str(p.relative_to(CLASSES)): digest(p) for p in CLASSES.rglob("*.class")}
    if classes != bindings["classes"]:
        raise ValueError("Verified private JVM derivative changed")
    boundary_path = CLASSES.parent / "boundary_match_verification.json"
    boundary = json.loads(boundary_path.read_text())
    if not (
        boundary["valid"]
        and boundary["exact_original_overload_comparisons"] == 113400
        and boundary["classes"] == classes
        and boundary["check_source_sha256"]
        == digest("casmi_ml/java/MetFragMassReuseCheck.java")
        and boundary["patcher_source_sha256"]
        == digest("casmi_ml/java/MetFragMassReusePatcher.java")
    ):
        raise ValueError("Exact source-bound boundary-match proof required")
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Original immutable CPU release changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    shutil.copytree(CLASSES, bundle / "mass_reuse_classes")
    for name in SOURCES:
        shutil.copy2("casmi_ml/" + name + ".py", bundle / "casmi_ml" / (name + ".py"))
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["mass_reuse_runtime"] = {
        "sources": {name: digest("casmi_ml/" + name + ".py") for name in SOURCES},
        "classes": classes,
        "full_rank_matches": {"unknown": 2000, "known": 1340},
        "all400_generation_csv_audit_fullhandoffs_exact": True,
        "all400_top25_exact": True,
        "all265_raw_physical_dictionaries_exact": True,
        "boundary_match_comparisons_exact": 113400,
        "boundary_match_proof_sha256": digest(boundary_path),
        "cold_control_sha256": digest(COLD / "report.json"),
        "scoring_control_sha256": digest(CONTROL / "report.json"),
        "control_sources_sha256": digest(CONTROL / "mass_reuse_sources.json"),
        "patcher_source_sha256": digest("casmi_ml/java/MetFragMassReusePatcher.java"),
        "boundary_check_source_sha256": digest(
            "casmi_ml/java/MetFragMassReuseCheck.java"
        ),
        "scope": "Original pinned jar/recipe/models/seeds/128samples/critic/peaks/candidates/scoring remain unchanged. A private classpath derivative adds the same matching method with explicit original ordered mass sum reused only within one unchanged fragment peak loop. Retained original methods instruction-identical except one calculate loop callsite, no persistent mass cache. Original60/1200/1800/8192 limits. Full3340/newcold400 local controls, separate frozen75/newfrozen400/platform400 proof required before unused singlecompetitionPOST. No query predictions/physics caches/labels/input-derived reference subset packaged.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    notebook = output / "notebook/casmi_chemistry.ipynb"
    document = json.loads(notebook.read_text())
    document["cells"][1]["source"] = document["cells"][1]["source"].replace(
        "casmi_ml.portable_verified_overlap_cpu_pipeline",
        "casmi_ml.portable_mass_reuse_pipeline",
    )
    write_json(notebook, document)
    dataset = "giaok246/casmi26-236-mass-reuse-best-bundle"
    metadata = json.loads((output / "notebook/kernel-metadata.json").read_text())
    metadata.update(
        id="giaok246/casmi26-236-mass-reuse-verified-best",
        title="CASMI26 236 Mass Reuse Verified Best",
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
    metadata.update(id=dataset, title="CASMI26 Scientific236 Mass Reuse Assets")
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
            "reason": "Same ordered fragment mass sum reused within each unchanged peak loop; no score/input/recipe/deadline change, cold400/3340 exact. Single unused explicit236 submission request; future localonly.",
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True)
    p.add_argument("--output", required=True)
    prepare(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
