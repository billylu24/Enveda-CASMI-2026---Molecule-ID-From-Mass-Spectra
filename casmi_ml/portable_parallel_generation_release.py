"""Freeze the original science assets with a source-bound whole-molecule scheduler."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path(
    "artifacts/research_loop/rounds/0376_parallel_generation_corrected_full400_control"
)


def prepare(old, output):
    old, output = Path(old), Path(output)
    if not Controller().local_only() or output.exists():
        raise ValueError("Fresh local-only runtime amendment required")
    proof = json.loads((CONTROL / "report.json").read_text())
    if (
        not proof["valid"]
        or proof["source_sha256"] != digest("casmi_ml/parallel_generation_runtime.py")
        or proof["original_generation_source_sha256"]
        != digest("casmi_ml/generation_inference.py")
        or not proof["full400_generation_handoffs_exact"]
        or not proof["all_generation_audits_exact"]
    ):
        raise ValueError(
            "Complete exact source-bound400 generation partition control required"
        )
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Original frozen reference assets changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in (
        "parallel_generation_runtime.py",
        "portable_parallel_generation_pipeline.py",
    ):
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["parallel_generation_runtime"] = {
        **proof,
        "verification_report_sha256": digest(CONTROL / "report.json"),
        "scope": "Complete original400 generation handoff and audit exact against independent sequential whole-molecule partition control. Same single-molecule forward graph/model/sample count/independent spectrum seed/insertion/deadlines. Use both assigned GPUs when available, oneGPU original path by default; no query prediction or physical cache packaged. New source-bound75/cold400/platform400 proof required.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    notebook = output / "notebook/casmi_chemistry.ipynb"
    n = json.loads(notebook.read_text())
    n["cells"][1]["source"] = n["cells"][1]["source"].replace(
        "casmi_ml.portable_reference_pipeline",
        "casmi_ml.portable_parallel_generation_pipeline",
    )
    write_json(notebook, n)
    dataset = "giaok246/casmi26-236-dualgen-best-bundle"
    metadata = output / "notebook/kernel-metadata.json"
    m = json.loads(metadata.read_text())
    m.update(
        id="giaok246/casmi26-236-dualgen-verified-best",
        title="CASMI26 236 Dualgen Verified Best",
        dataset_sources=[dataset, "aidensong123/casmi26-coconut-202609"],
        machine_shape="NvidiaTeslaT4",
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
    m.update(id=dataset, title="CASMI26 Scientific236 Dualgen Assets")
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
            "reason": "Reference platform64s retrieval+365s generation still exhausted original1200s physical deadline. Platform providesT4x2; independent original molecule-seeded generation partitions use both GPUs while all models/recipe/resource/physical/ranking gates stay unchanged. No new competition submission authorization.",
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
