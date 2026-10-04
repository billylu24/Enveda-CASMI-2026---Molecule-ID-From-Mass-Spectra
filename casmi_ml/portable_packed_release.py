"""Fresh exact storage runtime amendment after a platform budget failure."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_runtime import resolve
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT


def prepare(old, output):
    c = Controller()
    state = c.read()
    old, output = Path(old), Path(output)
    if not c.local_only() or output.exists():
        raise ValueError("Fresh local-only amendment required")
    if resolve(c, state["local_candidate_runtime"]["catalog_sha256"]) is None:
        raise ValueError("Fully verified object/packed local runtime required")
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Original verified source changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in [
        "portable_packed_pipeline.py",
        "packed_candidate_cache.py",
        "object_candidate_runtime.py",
    ]:
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    store = Path(state["local_candidate_runtime"]["store"])
    destination = bundle / "packed_fingerprints"
    destination.mkdir()
    for name in ["protocol.json", "report.json", "fingerprints.npy", "valid.npy"]:
        shutil.copy2(store / name, destination / name)
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["portable_packed_runtime"] = {
        "packed_source_sha256": digest("casmi_ml/packed_candidate_cache.py"),
        "object_source_sha256": digest("casmi_ml/object_candidate_runtime.py"),
        "full_rank_matches": {"unknown": 2000, "known": 1340},
        "packed_local_proof": state["local_candidate_runtime"],
        "object_local_proof": state["local_object_candidate_runtime"],
        "scope": "Pure molecular fingerprints derived from original catalog; no query-dependent or physical evidence cache. Original frozen science recipe/models unchanged; new cold400+fresh75+platform400 proof required.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    notebook_path = output / "notebook/casmi_chemistry.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"][1]["source"] = notebook["cells"][1]["source"].replace(
        "casmi_ml.portable_verified_pipeline", "casmi_ml.portable_packed_pipeline"
    )
    write_json(notebook_path, notebook)
    metadata = json.loads((output / "notebook/kernel-metadata.json").read_text())
    dataset_id = "giaok246/casmi26-236-packed-best-bundle"
    metadata.update(
        id="giaok246/casmi26-236-packed-verified-best",
        title="CASMI26 236 Packed Verified Best",
        dataset_sources=[dataset_id, "aidensong123/casmi26-coconut-202609"],
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
    metadata.update(
        id=dataset_id, title="CASMI26 Scientific236 Packed Inference Assets"
    )
    write_json(output / "dataset/dataset-metadata.json", metadata)
    identity = release_identity(output, sums, VARIANT)
    write_json(
        output / "status.json",
        {
            "status": "runtime_amendment_prepared",
            "identity": identity,
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
            "reason": "Initial platform exhausted the original1200-second fragmentation scene budget. Port previously all3340-exact packed/object molecular storage; keep budgets, models and science unchanged. Original rejected platform outputs retained.",
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--old", required=True, type=Path)
    p.add_argument("--output", required=True, type=Path)
    print(prepare(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
