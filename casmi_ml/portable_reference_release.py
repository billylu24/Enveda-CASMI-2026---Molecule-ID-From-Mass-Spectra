"""Source-qualified same-science release using complete query-independent train vectors."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path("artifacts/research_loop/rounds/0362_train_reference_cache_exact_subset")


def prepare(old, output, store):
    old, output, store = Path(old), Path(output), Path(store)
    if not Controller().local_only() or output.exists():
        raise ValueError("Fresh local-only amendment required")
    proof = json.loads((CONTROL / "report.json").read_text())
    if (
        not proof["valid"]
        or proof["cache_source_sha256"] != digest("casmi_ml/train_reference_cache.py")
        or proof["store_report_sha256"] != digest(store / "report.json")
    ):
        raise ValueError("Current exact full400 reference proof required")
    cache = json.loads((store / "report.json").read_text())
    if not cache["valid"] or cache["query_dependent"]:
        raise ValueError("Complete train-only vectors required")
    sums = json.loads((old / "bundle/SHA256SUMS.json").read_text())
    if any(digest(old / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Original release source changed")
    bundle = output / "bundle"
    shutil.copytree(
        old / "bundle", bundle, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    for name in ("train_reference_cache.py", "portable_reference_pipeline.py"):
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    (bundle / "train_reference_cache").mkdir()
    for name in ("protocol.json", "report.json", "rows.parquet", "spectra.npz"):
        shutil.copy2(store / name, bundle / "train_reference_cache" / name)
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["train_reference_runtime"] = {
        "source_sha256": digest("casmi_ml/train_reference_cache.py"),
        "verification_report_sha256": digest(CONTROL / "report.json"),
        "store_report_sha256": digest(store / "report.json"),
        "full400_rows_sparse_entries_and_members_exact": True,
        "train_sha256": cache["train_sha256"],
        "query_dependent_cache": False,
        "scope": "Only full original training library formula/vector/quality preprocessing; query-dependent subsets selected at runtime with original arithmetic. No packaged query predictions, mass subsets or physical scores. Same models/science/deadlines. Fresh75/cold400/platform400 remain required.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    shutil.copytree(old / "notebook", output / "notebook")
    p = output / "notebook/casmi_chemistry.ipynb"
    n = json.loads(p.read_text())
    n["cells"][1]["source"] = n["cells"][1]["source"].replace(
        "casmi_ml.portable_streaming_pipeline", "casmi_ml.portable_reference_pipeline"
    )
    write_json(p, n)
    dataset = "giaok246/casmi26-236-reference-best-bundle"
    p = output / "notebook/kernel-metadata.json"
    m = json.loads(p.read_text())
    m.update(
        id="giaok246/casmi26-236-reference-verified-best",
        title="CASMI26 236 Reference Verified Best",
        dataset_sources=[dataset, "aidensong123/casmi26-coconut-202609"],
    )
    write_json(p, m)
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
    ) as z:
        for name in [*sums, "SHA256SUMS.json"]:
            z.write(bundle / name, name)
    m = json.loads((old / "dataset/dataset-metadata.json").read_text())
    m.update(id=dataset, title="CASMI26 Scientific236 Train Reference Assets")
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
            "reason": "Streaming platform physical timeout near original shared1200s deadline. Reuse source-bound complete train-only vectors with full400 exact reference subset proof; all model/scientific/runtime budget gates retained.",
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("old", "output", "store"):
        p.add_argument("--" + n, required=True, type=Path)
    print(prepare(**vars(p.parse_args())))


if __name__ == "__main__":
    main()
