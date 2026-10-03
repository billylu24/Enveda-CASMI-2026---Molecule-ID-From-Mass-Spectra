"""Freeze the verified local scientific best for an explicitly requested release."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.local_verified_extension import verified_config
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.research_release import copy_inference_source
from casmi_ml.validation_cache_runtime import resolve as validation_runtime

VARIANT = "supported_leader_replay"


def prepare(output, base=Path("kaggle_release_strong_slots_0181_shared_v2")):
    c = Controller()
    source, proof = verified_config(c)
    if proof["scientific_round"] != "0236_low_supported_leader_first":
        raise ValueError("This release supports only the frozen236 scientific rule")
    if source.get("low_fragment") != {
        "supported_leader_first": True,
        "original_prefix": 3,
        "slots": 3,
    }:
        raise ValueError("Frozen236 low-fragment rule differs")
    if any(
        key in source
        for key in (
            "query_residual",
            "candidate_compatibility",
            "extended_candidate_compatibility",
        )
    ):
        raise ValueError("Unqualified experimental scores cannot enter this release")
    if not all(g["eligible"] for g in proof["gates"].values()):
        raise ValueError("Verified best comparator gates required")
    if validation_runtime(c) is None:
        raise ValueError("Current-source full validation-cache proof required")
    output, base = Path(output), Path(base)
    if output.exists():
        raise ValueError("Fresh release directory required")
    sums = json.loads((base / "bundle/SHA256SUMS.json").read_text())
    if any(digest(base / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Base release checksum changed")
    bundle = output / "bundle"
    shutil.copytree(
        base / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    copy_inference_source(bundle)
    recipe = json.loads((bundle / "deployment_recipe.json").read_text())
    config = dict(source)
    for name in ("catalog", "encoder", "critic", "prior", "jar"):
        target = recipe["external_routed"][name]
        if (
            digest(source[name]) != source[name + "_sha256"]
            or digest(bundle / target) != source[name + "_sha256"]
        ):
            raise ValueError("Frozen scientific asset differs: " + name)
        config[name] = target
    config["high_fragment"] = dict(source["high_fragment"])
    config["high_fragment"]["worker_class"] = "java_classes/MetFragWorker.class"
    runtime = dict(config["high_fragment"]["runtime"])
    runtime["process_class"] = recipe["external_routed"]["high_fragment"]["runtime"][
        "process_class"
    ]
    config["high_fragment"]["runtime"] = runtime
    for path, sha in [
        (
            config["high_fragment"]["worker_class"],
            config["high_fragment"]["worker_class_sha256"],
        ),
        (runtime["process_class"], runtime["process_class_sha256"]),
    ]:
        if digest(bundle / path) != sha:
            raise ValueError("Verified Java runtime asset changed")
    recipe.update(
        external_routed=config,
        scientific_round=proof["scientific_round"],
        status="development_experimental",
        independent_acceptance=False,
    )
    write_json(bundle / "deployment_recipe.json", recipe)
    decision = json.loads(
        Path("results/research_loop/0236_low_supported_leader_first.json").read_text()
    )
    decision["verified_evidence_amendment"] = proof
    write_json(bundle / "development_decision.json", decision)
    write_json(
        bundle / "release_evidence.json",
        {
            "verified_inference": proof,
            "validation_runtime": c.read()["local_validation_runtime"],
            "scope": "Frozen scientific236; portable exact array/identity/validation memoization only. No development physical caches or test truth packaged. New cold/platform verification required.",
        },
    )
    dataset_id = "giaok246/casmi26-236-verified-best-bundle"
    shutil.copytree(base / "notebook", output / "notebook")
    metadata = json.loads((output / "notebook/kernel-metadata.json").read_text())
    metadata.update(
        id="giaok246/casmi26-236-verified-best",
        title="CASMI26 236 Verified Local Best",
        dataset_sources=[dataset_id, "aidensong123/casmi26-coconut-202609"],
    )
    write_json(output / "notebook/kernel-metadata.json", metadata)
    notebook_path = output / "notebook/casmi_chemistry.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"][0]["source"] = (
        "# CASMI26 verified scientific236\nFrozen supported low-leader rule and strong reference slots. Repeated development; no independent accuracy acceptance. One explicitly requested competition submission.\n"
    )
    notebook["cells"][1]["source"] = notebook["cells"][1]["source"].replace(
        "casmi_ml.research_pipeline", "casmi_ml.portable_verified_pipeline"
    )
    write_json(notebook_path, notebook)
    sums = {
        str(p.relative_to(bundle)): digest(p)
        for p in bundle.rglob("*")
        if p.is_file()
        and p.name not in ("manifest.json", "SHA256SUMS.json")
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    write_json(bundle / "SHA256SUMS.json", sums)
    dataset = output / "dataset"
    dataset.mkdir()
    with zipfile.ZipFile(
        dataset / "casmi_chemistry_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in [*sums, "SHA256SUMS.json"]:
            archive.write(bundle / name, name)
    metadata = json.loads((base / "dataset/dataset-metadata.json").read_text())
    metadata.update(
        id=dataset_id,
        title="CASMI26 Verified Scientific236 Assets",
        description="Private frozen scientific236 inference assets; repeated development, no independent acceptance. ChEMBL attribution retained. No development physical caches or query labels.",
    )
    write_json(dataset / "dataset-metadata.json", metadata)
    identity = release_identity(output, sums, VARIANT)
    write_json(
        output / "status.json",
        {
            "status": "prepared",
            "identity": identity,
            "scientific_round": proof["scientific_round"],
            "development_eligible": True,
            "independent_acceptance": False,
            "uploaded": False,
            "submitted": False,
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    print(prepare(p.parse_args().output))


if __name__ == "__main__":
    main()
