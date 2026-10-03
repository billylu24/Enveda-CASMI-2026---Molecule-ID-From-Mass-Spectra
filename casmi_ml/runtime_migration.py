"""Fresh ABI releases: frozen models, identical local rankings, bound platform verification."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.experimental_release import package
from casmi_ml.research_loop import Controller, now


def prepare_round(identifier, output):
    output = Path(output)
    c = Controller()
    r = next(r for r in c.read()["rounds"] if r["id"] == identifier)
    if r["status"] != "eligible":
        raise ValueError("Only an unsubmitted eligible round can migrate")
    if output.exists():
        raise ValueError("Fresh runtime release directory required")
    old = r["release"]
    package(identifier + "_runtime313", r["decision"], output)
    write_json(
        output / "runtime_migration.json",
        {
            "round": identifier,
            "old_release": old,
            "new_release": str(output),
            "created_at": now(),
        },
    )
    return str(output)


def activate_round(identifier, output, old_release=None):
    output = Path(output)
    c = Controller()
    state = c.read()
    r = next(r for r in state["rounds"] if r["id"] == identifier)
    if r["status"] != "eligible":
        raise ValueError("Cannot replace an accepted competition submission")
    if str(output) == r["release"]:
        return r["runtime_amendment"]
    old = Path(
        old_release
        or json.loads((output / "runtime_migration.json").read_text())["old_release"]
    )
    if str(old) != r["release"]:
        raise ValueError("Migration source differs from current release")
    a, b = [pd.read_csv(p / "local_output/submission.csv") for p in (old, output)]
    if len(a) < 400 or not a.equals(b):
        raise ValueError("All local full rankings must match immutable original")
    osums, nsums = [
        json.loads((p / "bundle/SHA256SUMS.json").read_text()) for p in (old, output)
    ]
    models = {n: sha for n, sha in osums.items() if Path(n).suffix == ".pt"}
    if not models or any(nsums.get(n) != sha for n, sha in models.items()):
        raise ValueError("Frozen models changed")
    recipes = [
        json.loads((p / "bundle/deployment_recipe.json").read_text())
        for p in (old, output)
    ]
    if recipes[0] != recipes[1]:
        raise ValueError("Frozen recipe changed")
    v = json.loads((output / "verification.json").read_text())
    if not v["valid"] or v["molecules"] != len(b):
        raise ValueError("Complete inference validation required")
    v.update(
        previous_local_full_top25_matches=len(a),
        frozen_models_match=True,
        recipe_unchanged=True,
    )
    write_json(output / "verification.json", v)
    old_identity = json.loads((old / "status.json").read_text())["identity"]
    canonical_old_identity = state.get("submission_aliases", {}).get(
        old_identity, old_identity
    )
    previous = state["submissions"].get(canonical_old_identity)
    if previous and (
        previous.get("id") is not None
        or previous["status"] not in ("quota_wait", "superseded_runtime")
    ):
        raise ValueError("Never migrate accepted or ambiguous submission contents")
    aggregate = Path("results/research_loop") / f"{identifier}_runtime313.json"
    amendment = {
        "old_release": str(old),
        "new_release": str(output),
        "new_identity": v["identity"],
        "old_identity": old_identity,
        "changed_at": now(),
        "reason": "Dual ABI offline RDKit; frozen models and recipe unchanged; full local rankings identical",
        "aggregate": str(aggregate),
    }

    def activate(state):
        entry = next(row for row in state["rounds"] if row["id"] == identifier)
        if entry["release"] != str(old) or entry["status"] != "eligible":
            raise ValueError("Release changed while checking migration")
        current_canonical = state.get("submission_aliases", {}).get(
            old_identity, old_identity
        )
        previous = state["submissions"].get(current_canonical)
        if previous and (
            previous.get("id") is not None
            or previous["status"] not in ("quota_wait", "superseded_runtime")
        ):
            raise ValueError("Submission accepted or ambiguous during migration")
        if previous:
            previous.update(
                status="superseded_runtime",
                superseded_by=v["identity"],
                error="Unaccepted release replaced for runtime compatibility",
            )
        entry.update(
            release=str(output),
            remote_release=None,
            dataset_uploaded=False,
            identity=None,
            requires_platform_verification=True,
            git_synced=False,
            runtime_amendment=amendment,
        )

    c.change(activate)
    publish = Path("configs/research_publish_paths.json")
    paths = json.loads(publish.read_text())
    for name in [
        str(output / p)
        for p in [
            "notebook/casmi_chemistry.ipynb",
            "notebook/kernel-metadata.json",
            "verification.json",
            "status.json",
            "kaggle_verification.json",
        ]
    ]:
        if name not in paths["paths"]:
            paths["paths"].append(name)
    write_json(publish, paths)
    write_json(
        aggregate,
        {
            "round": identifier,
            "amendment": amendment,
            "frozen_models": models,
            "local_verification": v,
            "platform_verification_pending": True,
            "independent_acceptance": False,
        },
    )
    c.register_round(identifier + "_runtime313", "publication", [], str(aggregate))
    c.mark_round(identifier + "_runtime313", status="recorded", completed_at=now())
    return amendment


def verify_platform(identifier, output_dir):
    c = Controller()
    r = next(r for r in c.read()["rounds"] if r["id"] == identifier)
    release, output_dir = Path(r["release"]), Path(output_dir)
    a, b = [
        pd.read_csv(p / "submission.csv")
        for p in (release / "local_output", output_dir)
    ]
    if len(a) < 400 or not a.equals(b):
        raise ValueError("Platform full rankings differ from verified local result")
    report = json.loads((output_dir / "submission.csv.report.json").read_text())
    local = json.loads((release / "verification.json").read_text())
    local_report = json.loads(
        (release / "local_output/submission.csv.report.json").read_text()
    )
    binding_fields = [
        "checkpoint_sha256",
        "baseline_sha256",
        "critic_checkpoint_sha256",
        "prefix",
        "slots",
        "open_protected",
        "frequency_weight",
        "token_length_exponent",
        "critic_weight",
        "adaptive_prefix",
        "expanded_prefix",
        "first_gate",
        "sampling",
        "formula_oracle_used",
    ]
    if any(report.get(k) != local_report.get(k) for k in binding_fields):
        raise ValueError(
            "Platform checkpoint or inference rule differs from local package"
        )
    if local_report.get("external_routed"):
        external_local, external_remote = (
            local_report["external_routed"],
            report.get("external_routed", {}),
        )
        if any(
            external_local.get(key) != external_remote.get(key)
            for key in ("config", "source_sha256", "unlabeled_input")
        ):
            raise ValueError("Platform external candidate rules or source differ")
    if (
        report["seconds"] > c.config["inference_seconds"]
        or report["peak_rss_mib"] > c.config["inference_rss_mib"]
    ):
        raise ValueError("Platform resource gate failed")
    result = {
        "valid": True,
        "identity": local["identity"],
        "molecules": len(b),
        "local_full_top25_matches": len(b),
        "seconds": report["seconds"],
        "parent_peak_rss_mib": report["peak_rss_mib"],
        "remote_release": r["remote_release"],
        "independent_acceptance": False,
        "inference_binding_fields": binding_fields,
        "scope": "Visible full rankings and resources; original development and replay gates retained; no new accuracy evidence",
    }
    write_json(release / "kaggle_verification.json", result)
    status = json.loads((release / "status.json").read_text())
    status.update(
        status="platform_verified_quota_wait",
        uploaded=True,
        dataset_uploaded=True,
        kernel=r["remote_release"]["kernel"],
        kernel_version=r["remote_release"]["version"],
    )
    write_json(release / "status.json", status)
    amendment = r.get("runtime_amendment", {})
    if amendment.get("aggregate"):
        path = Path(amendment["aggregate"])
        aggregate = json.loads(path.read_text())
        aggregate.update(
            platform_verification_pending=False, platform_verification=result
        )
        write_json(path, aggregate)
    c.mark_round(identifier, git_synced=False)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("prepare", "activate", "verify-platform"))
    p.add_argument("--round", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--old-release", type=Path)
    a = p.parse_args()
    fn = {
        "prepare": prepare_round,
        "activate": activate_round,
        "verify-platform": verify_platform,
    }[a.command]
    result = (
        fn(a.round, a.output, a.old_release)
        if a.command == "activate"
        else fn(a.round, a.output)
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
