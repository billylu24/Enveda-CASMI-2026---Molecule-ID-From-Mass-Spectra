"""Package the development-gated, replayed ChEMBL routed extension privately."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.data import write_json
from casmi_ml.experimental_release import publication_slug
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import release_identity
from casmi_ml.research_release import copy_inference_source


def prepare(
    directory,
    output,
    base=Path("kaggle_release_gate_0062_runtime313"),
    shared_reference=False,
):
    directory, output, base = Path(directory), Path(output), Path(base)
    decision = json.loads((directory / "decision.json").read_text())
    if (
        not decision["winner"]
        or not decision["winner"]["gate"]["eligible"]
        or decision["direction"]
        not in (
            "chembl_routed_combination",
            "chembl_high_fragment",
            "chembl_high_strong_slots",
        )
    ):
        raise ValueError("Frozen routed-combination development gate required")
    verification = json.loads((directory / "replay.json").read_text())
    if not verification["valid"] or verification["molecules"] < 50:
        raise ValueError("Actual unlabeled external replay required")
    if decision["direction"] == "chembl_high_strong_slots":
        if (
            verification.get("full_rank_matches") != verification["molecules"]
            or not verification.get("unlabeled_input")
            or verification.get("external_source_sha256")
            != digest("casmi_ml/chembl_routed_inference.py")
            or verification.get("protocol_sha256")
            != digest(directory / "protocol.json")
        ):
            raise ValueError("Exact source-bound181 replay required")
        source = json.loads((directory / "external_config.json").read_text())
        if source.get("high_fragment", {}).get("strong_slots") != {
            "reference_threshold": 0.5,
            "original_prefix": 3,
            "slots": 3,
        }:
            raise ValueError("Frozen181 strong slot recipe required")
    if output.exists():
        raise ValueError("Fresh release directory required")
    output.mkdir(parents=True)
    bundle = output / "bundle"
    shutil.copytree(
        base / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    copy_inference_source(bundle)
    recipe = json.loads((bundle / "deployment_recipe.json").read_text())
    if recipe["generation"]["first_gate"] != {"confidence": 0.2, "margin": 0.05}:
        raise ValueError("Frozen0062 baseline generation gate differs")
    source = json.loads((directory / "external_config.json").read_text())
    names = {
        "catalog": "chembl_catalog.parquet",
        "encoder": "model.pt",
        "critic": "generated_critic.pt",
        "prior": "fingerprint_prior.npy",
        "jar": "metfrag.jar",
    }
    config = {}
    for name, target in names.items():
        src = Path(source[name])
        dest = bundle / target
        if digest(src) != source[name + "_sha256"]:
            raise ValueError("External source changed")
        if not dest.exists() or digest(dest) != source[name + "_sha256"]:
            shutil.copy2(src, dest)
        config[name] = target
        config[name + "_sha256"] = digest(dest)
    if source.get("high_fragment"):
        high = source["high_fragment"]
        if (
            high.get("prefix") != 3
            or high.get("backend") != "persistent"
            or high.get("weight") != 0.5
        ):
            raise ValueError("Unsupported high fragment release rule")
        worker_src = Path(high["worker_class"])
        if digest(worker_src) != high["worker_class_sha256"]:
            raise ValueError("Java worker checksum changed")
        worker_dest = bundle / "java_classes/MetFragWorker.class"
        worker_dest.parent.mkdir()
        shutil.copy2(worker_src, worker_dest)
        worker_source = Path("casmi_ml/java/MetFragWorker.java")
        if digest(worker_source) != high["worker_source_sha256"]:
            raise ValueError("Java worker source changed")
        shutil.copy2(worker_source, worker_dest.parent / "MetFragWorker.java")
        config["high_fragment"] = dict(
            high, worker_class="java_classes/MetFragWorker.class"
        )
        if high.get("runtime"):
            from casmi_ml.metfrag_polling_patch import CLASS_NAME, PATCHED_SHA

            runtime = high["runtime"]
            process_source = Path(runtime["process_class"])
            if (
                digest(process_source) != PATCHED_SHA
                or runtime["process_class_sha256"] != PATCHED_SHA
            ):
                raise ValueError("Verified derived runtime process required")
            process = worker_dest.parent / CLASS_NAME
            process.parent.mkdir(parents=True)
            shutil.copy2(process_source, process)
            config["high_fragment"]["runtime"] = dict(
                runtime, process_class=str(process.relative_to(bundle))
            )
        if (
            high.get("strong_slots")
            != {"reference_threshold": 0.5, "original_prefix": 3, "slots": 3}
            and decision["direction"] == "chembl_high_strong_slots"
        ):
            raise ValueError("Frozen181 strong slot recipe required")
    attribution = bundle / "chembl_attribution"
    attribution.mkdir()
    for name in (
        "LICENSE",
        "README",
        "REQUIRED.ATTRIBUTION",
        "ATTRIBUTION.md",
        "manifest.json",
    ):
        if (DERIVED.parent / name).is_file():
            shutil.copy2(DERIVED.parent / name, attribution / name)
    if shared_reference:
        if decision["direction"] != "chembl_high_strong_slots":
            raise ValueError("Shared reference only supports181")
        recipe["reference_runtime"] = {"scan": "shared_union_v1"}
    recipe["external_routed"] = config
    recipe["development_decision"] = "development_decision.json"
    recipe["independent_acceptance"] = False
    recipe["status"] = "development_experimental"
    write_json(bundle / "deployment_recipe.json", recipe)
    shutil.copy2(directory / "decision.json", bundle / "development_decision.json")
    sums = {
        str(p.relative_to(bundle)): digest(p)
        for p in bundle.rglob("*")
        if p.is_file()
        and p.name not in ["manifest.json", "SHA256SUMS.json"]
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
    slug = publication_slug(directory.name)
    dataset_id = f"giaok246/casmi26-research-{slug}-bundle"
    kernel_id = f"giaok246/casmi26-research-{slug}"
    metadata = json.loads((base / "dataset/dataset-metadata.json").read_text())
    metadata.update(
        id=dataset_id,
        title=f"CASMI Research {slug} Assets",
        description="Private frozen0062 plus development-gated ChEMBL routed candidates; repeated development, no independent acceptance. ChEMBL CC-BY-SA3.0 attribution included.",
    )
    write_json(output / "dataset/dataset-metadata.json", metadata)
    shutil.copytree(base / "notebook", output / "notebook")
    kernel = json.loads((output / "notebook/kernel-metadata.json").read_text())
    kernel.update(
        id=kernel_id,
        title=f"CASMI26 Research {slug}",
        dataset_sources=[dataset_id, "aidensong123/casmi26-coconut-202609"],
    )
    write_json(output / "notebook/kernel-metadata.json", kernel)
    notebook = json.loads((output / "notebook/casmi_chemistry.ipynb").read_text())
    notebook["cells"][0]["source"] = (
        f"# CASMI Research {directory.name}\nFrozen0062 plus routed ChEMBL candidates. Development-gated experiment; repeated development, no independent acceptance.\n"
    )
    write_json(output / "notebook/casmi_chemistry.ipynb", notebook)
    identity = release_identity(output, sums, decision["winner"]["variant"])
    write_json(
        output / "status.json",
        {
            "status": "prepared",
            "identity": identity,
            "development_eligible": True,
            "independent_acceptance": False,
            "uploaded": False,
            "submitted": False,
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--shared-reference", action="store_true")
    a = p.parse_args()
    print(prepare(a.directory, a.output, shared_reference=a.shared_reference))


if __name__ == "__main__":
    main()
