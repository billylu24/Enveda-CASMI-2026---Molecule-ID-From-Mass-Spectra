"""Build an offline bundle while preserving independent acceptance status."""

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.candidate_expansion import CATALOG, ENCODER, EXTERNAL, ROOT
from casmi_ml.data import write_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experimental-positive-points", action="store_true")
    args = parser.parse_args()
    report = json.loads((ROOT / "fresh_report.json").read_text())
    experimental = (
        args.experimental_positive_points
        and report.get("unknown", {}).get("difference", {}).get("difference", 0) > 0
        and report.get("known", {}).get("difference", {}).get("difference", -1) >= 0
    )
    if not report["accepted"] and not experimental:
        raise SystemExit("Acceptance failed; no release created")
    selection = json.loads((ROOT / "selection.json").read_text())
    release = Path("kaggle_release_expansion")
    bundle = release / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    original = Path("kaggle_release_scale/bundle")
    for source in original.rglob("*"):
        if (
            not source.is_file()
            or "__pycache__" in source.parts
            or source.suffix == ".pyc"
        ):
            continue
        target = bundle / source.relative_to(original)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for name in ["secondary_inference.py", "candidate_catalog.py"]:
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    shutil.copy2(CATALOG, bundle / "pubchemlite.parquet")
    shutil.copy2(EXTERNAL / "ATTRIBUTION.md", bundle / "PUBCHEMLITE_ATTRIBUTION.md")
    shutil.copy2(EXTERNAL / "manifest.json", bundle / "pubchemlite_manifest.json")
    shutil.copy2(
        EXTERNAL / "zenodo_record.json", bundle / "pubchemlite_source_record.json"
    )
    for name in [
        "protocol.json",
        "selection.json",
        "fresh_report.json",
        "verification.json",
        "REPORT.md",
    ]:
        shutil.copy2(ROOT / name, bundle / name)
    recipe = json.loads(
        Path("artifacts/scale_20260929/deployment_recipe.json").read_text()
    )
    recipe.update(
        status=(
            "passed_independent_candidate_expansion_acceptance"
            if report["accepted"]
            else "experimental_positive_point_estimates_statistical_gate_failed"
        ),
        checkpoint="model.pt",
        candidate_expansion={
            "path": "pubchemlite.parquet",
            "sha256": hashlib.sha256(CATALOG.read_bytes()).hexdigest(),
            "weight": selection["winner"]["weight"],
        },
    )
    recipe["config"]["name"] = "residual_60k_pubchemlite_fixed_historical_route"
    write_json(bundle / "recipe.json", recipe)
    local = {
        **recipe,
        "checkpoint": str(ENCODER.resolve()),
        "candidate_expansion": {
            **recipe["candidate_expansion"],
            "path": str(CATALOG.resolve()),
        },
    }
    write_json(ROOT / "deployment_recipe.json", local)
    sums = {
        str(p.relative_to(bundle)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(bundle.rglob("*"))
        if p.is_file()
        and p.name != "SHA256SUMS.json"
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    write_json(bundle / "SHA256SUMS.json", sums)
    dataset = release / "dataset"
    dataset.mkdir(exist_ok=True)
    with zipfile.ZipFile(
        dataset / "casmi_expansion_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sorted(bundle.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                archive.write(p, p.relative_to(bundle))
    write_json(
        dataset / "dataset-metadata.json",
        {
            "title": "CASMI 2026 PubChemLite Expansion Bundle",
            "id": "giaok246/casmi-2026-pubchemlite-expansion-bundle",
            "licenses": [{"name": "other"}],
        },
    )
    notebook = release / "notebook"
    notebook.mkdir(exist_ok=True)
    nb = json.loads(Path("kaggle_release_scale/notebook/casmi_scale.ipynb").read_text())
    nb["cells"][0]["source"] = [
        "# CASMI PubChemLite Expansion\n",
        "Frozen 60K residual model with an independent public candidate catalog. PubChemLite 2026-09-25: https://doi.org/10.5281/zenodo.22953038 (CC BY 4.0); attribution and modifications in PUBCHEMLITE_ATTRIBUTION.md. High-confidence historical retrieval preserved.\n",
    ]
    nb["cells"][1]["source"] = [
        line.replace("casmi_scale_bundle.zip", "casmi_expansion_bundle.zip")
        for line in nb["cells"][1]["source"]
    ]
    write_json(notebook / "casmi_expansion.ipynb", nb)
    metadata = json.loads(
        Path("kaggle_release_scale/notebook/kernel-metadata.json").read_text()
    )
    metadata.update(
        id="giaok246/casmi-2026-pubchemlite-expansion-inference",
        title="CASMI 2026 PubChemLite Expansion Inference",
        code_file="casmi_expansion.ipynb",
        dataset_sources=[
            "giaok246/casmi-2026-pubchemlite-expansion-bundle",
            "aidensong123/casmi26-coconut-202609",
        ],
    )
    write_json(notebook / "kernel-metadata.json", metadata)
    write_json(
        release / "status.json",
        {
            "status": "prepared_locally",
            "uploaded": False,
            "submitted": False,
            "local_holdout_accepted": report["accepted"],
            "experimental_submission": not report["accepted"],
            "submission_basis": "User authorized upload when local metrics improve; both independent point estimates improved. Statistical gate status remains unchanged.",
            "external_sha256": recipe["candidate_expansion"]["sha256"],
        },
    )
    print("Prepared", release)


if __name__ == "__main__":
    main()
