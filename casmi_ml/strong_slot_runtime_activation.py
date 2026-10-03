"""Adopt181 shared reference runtime only after exact fresh/local evidence."""

import argparse
import json
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity

ALLOWED = {
    "casmi_ml/chembl_routed_release.py",
    "casmi_ml/reference_reuse.py",
    "casmi_ml/research_loop.py",
    "casmi_ml/secondary_inference.py",
    "casmi_ml/shared_reference_benchmark.py",
    "deployment_recipe.json",
}


def validate(old, new, config):
    old, new = Path(old), Path(new)
    evidence = json.loads((new / "implementation_replay.json").read_text())
    if (
        not evidence.get("valid")
        or evidence.get("full_rank_matches") != 75
        or not evidence.get("fresh_fragment_cache")
    ):
        raise ValueError("Exact fresh75 required")
    sums = []
    for release in [old, new]:
        mapping = json.loads((release / "bundle/SHA256SUMS.json").read_text())
        if any(
            digest(release / "bundle" / name) != sha for name, sha in mapping.items()
        ):
            raise ValueError("Frozen bundle changed")
        sums.append(mapping)
        verification = json.loads((release / "verification.json").read_text())
        identity = release_identity(release, mapping, "strong_slot_prefix")
        if (
            not verification.get("valid")
            or verification.get("identity") != identity
            or verification.get("molecules") != 400
            or verification.get("seconds", float("inf")) > config["inference_seconds"]
            or verification.get("peak_rss_mib", float("inf"))
            > config["inference_rss_mib"]
        ):
            raise ValueError("Bound cold400 and resource gate required")
    if evidence["bundle_sums_sha256"] != digest(new / "bundle/SHA256SUMS.json"):
        raise ValueError("Fresh replay belongs to different bundle")
    changed = {
        name
        for name in set(sums[0]) | set(sums[1])
        if sums[0].get(name) != sums[1].get(name)
    }
    if changed - ALLOWED:
        raise ValueError("Scientific asset outside allowed runtime amendment changed")
    recipes = [
        json.loads((p / "bundle/deployment_recipe.json").read_text())
        for p in [old, new]
    ]
    if (
        recipes[1].pop("reference_runtime", None) != {"scan": "shared_union_v1"}
        or recipes[0] != recipes[1]
    ):
        raise ValueError("Scientific recipe changed")
    if recipes[0].get("external_routed", {}).get("high_fragment", {}).get(
        "strong_slots"
    ) != {"reference_threshold": 0.5, "original_prefix": 3, "slots": 3}:
        raise ValueError("Frozen181 rules required")
    frames = [pd.read_csv(p / "local_output/submission.csv") for p in [old, new]]
    if len(frames[0]) != 400 or not frames[0].equals(frames[1]):
        raise ValueError("All400 full rankings must be exact")
    return {
        "valid": True,
        "molecules": 400,
        "full_rank_matches": 400,
        "changed_runtime_files": sorted(changed),
        "new_identity": release_identity(new, sums[1], "strong_slot_prefix"),
        "old_identity": release_identity(old, sums[0], "strong_slot_prefix"),
        "science_recipe_equal": True,
        "model_assets_equal": True,
        "independent_acceptance": False,
    }


def activate(old, new):
    c = Controller()
    new = Path(new)
    rid = "0181_high_strong_slot_full"
    proof = validate(old, new, c.config)
    r = next(r for r in c.read()["rounds"] if r["id"] == rid)
    if r.get("remote_release") or r.get("dataset_uploaded"):
        raise ValueError("Cannot replace an already uploaded181 package")
    if (
        proof["old_identity"] in c.read()["submissions"]
        or proof["new_identity"] in c.read()["submissions"]
    ):
        raise ValueError("Cannot replace a reserved or submitted identity")
    write_json(new / "runtime_migration.json", proof)
    c.mark_round(
        rid,
        status="eligible",
        release=str(new),
        requires_platform_verification=True,
        runtime_amendment={
            "aggregate": "results/research_loop/strong_slot_shared_runtime_0181.json"
        },
        git_synced=False,
    )
    write_json(
        "results/research_loop/strong_slot_shared_runtime_0181.json",
        dict(
            proof,
            platform_verification_pending=True,
            scope="Same181 scientific recipe and models, shared union reference scan, fresh75 exact and cold400 exact; platform400 still required.",
        ),
    )
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--new", type=Path, required=True)
    print(json.dumps(activate(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
