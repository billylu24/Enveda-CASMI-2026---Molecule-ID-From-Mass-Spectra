"""Prepare an offline release only after frozen direct-ranking acceptance passes."""

import hashlib
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.direct_experiment import ENCODER, ROOT


def main():
    report = json.loads((ROOT / "fresh_report.json").read_text())
    if not report["accepted"]:
        raise SystemExit("Acceptance failed; no release created")
    selection = json.loads((ROOT / "selection.json").read_text())["winner"]
    release = Path("kaggle_release_direct")
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
    for name in ["secondary_inference.py", "direct_models.py"]:
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    shutil.copy2(ENCODER, bundle / "model.pt")
    shutil.copy2(selection["checkpoint"], bundle / "direct.pt")
    for name in [
        "protocol.json",
        "selection.json",
        "fresh_report.json",
        "verification.json",
    ]:
        shutil.copy2(ROOT / name, bundle / name)
    recipe = json.loads(
        Path("artifacts/scale_20260929/deployment_recipe.json").read_text()
    )
    recipe.update(
        status="passed_direct_ranking_independent_acceptance",
        checkpoint="model.pt",
        direct_ranker={
            "path": "direct.pt",
            "sha256": selection["checkpoint_sha256"],
            "weight": selection["weight"],
            "architecture": selection["architecture"],
        },
    )
    recipe["config"]["name"] = "residual_60k_direct_graph_historical_fixed_route"
    write_json(bundle / "recipe.json", recipe)
    local = {
        **recipe,
        "checkpoint": str(ENCODER.resolve()),
        "direct_ranker": {
            **recipe["direct_ranker"],
            "path": str(Path(selection["checkpoint"]).resolve()),
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
        dataset / "casmi_direct_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for p in sorted(bundle.rglob("*")):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc":
                archive.write(p, p.relative_to(bundle))
    write_json(
        dataset / "dataset-metadata.json",
        {
            "title": "CASMI 2026 Direct Graph Ranking Bundle",
            "id": "giaok246/casmi-2026-direct-graph-ranking-bundle",
            "licenses": [{"name": "other"}],
        },
    )
    notebook = release / "notebook"
    notebook.mkdir(exist_ok=True)
    nb = json.loads(Path("kaggle_release_scale/notebook/casmi_scale.ipynb").read_text())
    nb["cells"][0]["source"] = [
        "# CASMI Direct Graph Ranking\n",
        "Frozen residual spectrum encoder and mass-conditioned contrastive graph candidate ranking; independent single-query acceptance; fixed historical threshold 0.5.\n",
    ]
    nb["cells"][1]["source"] = [
        line.replace("casmi_scale_bundle.zip", "casmi_direct_bundle.zip")
        for line in nb["cells"][1]["source"]
    ]
    write_json(notebook / "casmi_direct.ipynb", nb)
    metadata = json.loads(
        Path("kaggle_release_scale/notebook/kernel-metadata.json").read_text()
    )
    metadata.update(
        id="giaok246/casmi-2026-direct-graph-inference",
        title="CASMI 2026 Direct Graph Inference",
        code_file="casmi_direct.ipynb",
        dataset_sources=[
            "giaok246/casmi-2026-direct-graph-ranking-bundle",
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
            "local_holdout_accepted": True,
            "architecture": selection["architecture"],
            "checkpoint_sha256": selection["checkpoint_sha256"],
        },
    )
    print("Prepared", release)


if __name__ == "__main__":
    main()
