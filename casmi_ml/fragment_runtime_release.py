"""Prepare an isolated163 runtime amendment; activate only after unchanged rankings."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.experimental_release import publication_slug
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_polling_patch import CLASS_NAME, PATCHED_SHA, PROCESS_SHA
from casmi_ml.research_loop import release_identity


def prepare(old, output):
    old, output = Path(old), Path(output)
    if output.exists():
        raise ValueError("Fresh runtime package required")
    verification = json.loads((old / "verification.json").read_text())
    if not verification["valid"] or verification["molecules"] != 400:
        raise ValueError("Original full400 verification required")
    decision = json.loads((old / "bundle/development_decision.json").read_text())
    if (
        not decision["winner"]["gate"]["eligible"]
        or decision["direction"] != "chembl_high_fragment"
    ):
        raise ValueError("Original high-fragment development gate required")
    reports = {}
    for identifier in (
        "0172_metfrag_threads2_benchmark",
        "0173_metfrag_polling_benchmark",
    ):
        report = json.loads(
            (
                Path("artifacts/research_loop/rounds") / identifier / "report.json"
            ).read_text()
        )
        if (
            report["exact_score_dictionary_matches"] != 12
            or not report["fresh_cache_per_arm"]
            or any(
                s.get("timeout", 0) or s.get("failed", 0)
                for s in report["spectrum_status_counts"].values()
            )
        ):
            raise ValueError("Real fresh exact score benchmarks required")
        reports[identifier] = report
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Immutable original release changed")
    output.mkdir(parents=True)
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in (
        "chembl_routed_inference.py",
        "metfrag_persistent.py",
        "metfrag_polling_patch.py",
    ):
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    patched = Path("artifacts/research_loop/metfrag_polling_classes") / CLASS_NAME
    if digest(patched) != PATCHED_SHA:
        raise ValueError("Verified derived runtime class changed")
    process = bundle / "java_classes" / CLASS_NAME
    process.parent.mkdir(parents=True)
    shutil.copy2(patched, process)
    recipe = json.loads((bundle / "deployment_recipe.json").read_text())
    recipe["external_routed"]["high_fragment"]["runtime"] = {
        "candidate_threads": 2,
        "polling_ms": 10,
        "process_class": str(process.relative_to(bundle)),
        "process_class_sha256": PATCHED_SHA,
        "original_process_class_sha256": PROCESS_SHA,
    }
    write_json(bundle / "deployment_recipe.json", recipe)
    write_json(
        bundle / "metfrag_attribution/runtime_change.json",
        {
            "upstream": "https://github.com/ipb-halle/MetFragRelaunched/releases/tag/v2.6.11",
            "jar_unchanged": True,
            "original_process_class_sha256": PROCESS_SHA,
            "derived_process_class_sha256": PATCHED_SHA,
            "change": "Only candidate executor completion polling1000ms->10ms; all other class bytes unchanged. NumberThreads2. Original fragmentation and score algorithms retained.",
            "patch_source": "casmi_ml/metfrag_polling_patch.py",
        },
    )
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
    slug = publication_slug("0163_fragment_runtime2_poll10")
    dataset = json.loads((old / "dataset/dataset-metadata.json").read_text())
    dataset.update(
        id=f"giaok246/casmi26-research-{slug}-bundle",
        title=f"CASMI Research {slug} Assets",
    )
    write_json(output / "dataset/dataset-metadata.json", dataset)
    shutil.copytree(old / "notebook", output / "notebook")
    kernel = json.loads((output / "notebook/kernel-metadata.json").read_text())
    kernel.update(
        id=f"giaok246/casmi26-research-{slug}",
        title=f"CASMI26 Research {slug}",
        dataset_sources=[dataset["id"], "aidensong123/casmi26-coconut-202609"],
    )
    write_json(output / "notebook/kernel-metadata.json", kernel)
    notebook_path = output / "notebook/casmi_chemistry.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"][0]["source"] = (
        "# CASMI163 experimental runtime verification\nFrozen163 ranking rules/models; candidate threads2 and executor completion polling10ms. Exact scores and full local/platform rankings required; no new accuracy claim.\n"
    )
    write_json(notebook_path, notebook)
    identity = release_identity(output, sums, decision["winner"]["variant"])
    write_json(
        output / "status.json",
        {
            "status": "runtime_prepared_not_activated",
            "identity": identity,
            "development_eligible": True,
            "independent_acceptance": False,
            "uploaded": False,
            "submitted": False,
        },
    )
    write_json(
        output / "runtime_migration.json",
        {
            "round": decision["round"],
            "old_release": str(old),
            "new_release": str(output),
            "benchmarks": reports,
            "science_recipe_change": False,
            "new_accuracy_claim": False,
            "required_next": "Fresh75 complete rankings and cold400 complete Top25 must match original before activation; platform full400 required before competition.",
        },
    )
    return output


def activate(identifier, output):
    import pandas as pd

    from casmi_ml.research_loop import Controller, now

    c = Controller()
    output = Path(output)
    migration = json.loads((output / "runtime_migration.json").read_text())
    old = Path(migration["old_release"])
    entry = next(r for r in c.read()["rounds"] if r["id"] == identifier)
    if entry["status"] != "eligible" or entry["release"] != str(old):
        raise ValueError("Only the current unsubmitted eligible release can migrate")
    replay = json.loads((output / "implementation_replay.json").read_text())
    verification = json.loads((output / "verification.json").read_text())
    if (
        not replay["valid"]
        or replay["full_rank_matches"] != 75
        or not replay["fresh_fragment_cache"]
        or not verification["valid"]
        or verification["molecules"] != 400
        or verification["seconds"] > c.config["inference_seconds"]
        or verification["peak_rss_mib"] > c.config["inference_rss_mib"]
    ):
        raise ValueError("Fresh75 and cold400 runtime/resource proofs required")
    old_sums, new_sums = [
        json.loads((p / "bundle/SHA256SUMS.json").read_text()) for p in (old, output)
    ]
    if digest(output / "bundle/SHA256SUMS.json") != replay["bundle_sums_sha256"]:
        raise ValueError("Runtime bundle changed after implementation replay")
    for release, sums in ((old, old_sums), (output, new_sums)):
        if any(digest(release / "bundle" / name) != sha for name, sha in sums.items()):
            raise ValueError("Verified bundle content changed")
    allowed = {
        "casmi_ml/chembl_routed_inference.py",
        "casmi_ml/metfrag_persistent.py",
        "casmi_ml/metfrag_polling_patch.py",
        "deployment_recipe.json",
        "java_classes/" + CLASS_NAME,
        "metfrag_attribution/runtime_change.json",
    }
    changes = {
        name
        for name in set(old_sums) | set(new_sums)
        if old_sums.get(name) != new_sums.get(name)
    }
    if not changes.issubset(allowed):
        raise ValueError("Non-runtime assets changed")
    if not any(Path(name).suffix == ".pt" for name in old_sums):
        raise ValueError("Frozen model assets required")
    recipes = [
        json.loads((p / "bundle/deployment_recipe.json").read_text())
        for p in (old, output)
    ]
    runtime = recipes[1]["external_routed"]["high_fragment"].pop("runtime")
    if (
        recipes[0] != recipes[1]
        or runtime["candidate_threads"] != 2
        or runtime["polling_ms"] != 10
    ):
        raise ValueError("Frozen scientific recipe differs")
    a, b = [pd.read_csv(p / "local_output/submission.csv") for p in (old, output)]
    if len(a) != 400 or not a.equals(b):
        raise ValueError("All400 complete Top25 must match immutable original")
    decision = json.loads((output / "bundle/development_decision.json").read_text())
    if (
        not decision["winner"]["gate"]["eligible"]
        or decision["direction"] != "chembl_high_fragment"
    ):
        raise ValueError("Original development gate required")
    if (
        release_identity(output, new_sums, decision["winner"]["variant"])
        != verification["identity"]
    ):
        raise ValueError("Cold verification identity differs")
    old_identity = json.loads((old / "status.json").read_text())["identity"]
    canonical = c.read().get("submission_aliases", {}).get(old_identity, old_identity)
    amendment = {
        "old_release": str(old),
        "new_release": str(output),
        "old_identity": old_identity,
        "new_identity": verification["identity"],
        "old_remote_release": entry.get("remote_release"),
        "reason": "Same frozen163 ranking rules/models and400 exact local ranks; verified threads2+10ms runtime only",
        "changed_at": now(),
        "aggregate": f"results/research_loop/{identifier}_fastpoll_runtime.json",
    }

    def update(state):
        current = next(r for r in state["rounds"] if r["id"] == identifier)
        if current["status"] != "eligible" or current["release"] != str(old):
            raise ValueError("Concurrent release change")
        previous = state["submissions"].get(canonical)
        if previous and (
            previous.get("id") is not None
            or previous["status"] not in ("quota_wait", "superseded_runtime")
        ):
            raise ValueError(
                "Never migrate accepted or ambiguous competition submission"
            )
        if previous:
            previous.update(
                status="superseded_runtime", superseded_by=verification["identity"]
            )
        current.update(
            release=str(output),
            remote_release=None,
            dataset_uploaded=False,
            identity=None,
            requires_platform_verification=True,
            runtime_amendment=amendment,
            git_synced=False,
        )

    c.change(update)
    verification.update(
        previous_local_full_top25_matches=400,
        frozen_models_match=True,
        science_recipe_unchanged=True,
        implementation_replay_matches=75,
    )
    write_json(output / "verification.json", verification)
    aggregate = {
        "round": identifier,
        "amendment": amendment,
        "local_verification": verification,
        "implementation_replay": replay,
        "changed_paths": sorted(changes),
        "platform_verification_pending": True,
        "independent_acceptance": False,
        "new_accuracy_claim": False,
    }
    write_json(amendment["aggregate"], aggregate)
    paths = Path("configs/research_publish_paths.json")
    public = json.loads(paths.read_text())
    for name in (
        "notebook/casmi_chemistry.ipynb",
        "notebook/kernel-metadata.json",
        "status.json",
        "verification.json",
        "implementation_replay.json",
        "kaggle_verification.json",
    ):
        value = str(output / name)
        if value not in public["paths"]:
            public["paths"].append(value)
    write_json(paths, public)
    c.register_round(
        identifier + "_fastpoll_runtime", "publication", [], amendment["aggregate"]
    )
    c.mark_round(identifier + "_fastpoll_runtime", status="recorded")
    return amendment


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--activate", action="store_true")
    p.add_argument("--round")
    a = p.parse_args()
    if a.activate:
        if not a.round:
            p.error("--round required to activate")
        print(activate(a.round, a.output))
    else:
        if not a.old:
            p.error("--old required to prepare")
        print(prepare(a.old, a.output))


if __name__ == "__main__":
    main()
