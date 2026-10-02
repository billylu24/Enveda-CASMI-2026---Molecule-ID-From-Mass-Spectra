"""Materialize a reviewable chemistry recipe only after independent acceptance."""

import argparse
import json
import shutil
from pathlib import Path

from casmi_ml.chemistry import VERSION
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, freeze


def copy_inference_source(output):
    output = Path(output)
    package = output / "casmi_ml"
    package.mkdir(parents=True, exist_ok=True)
    for source in Path("casmi_ml").glob("*.py"):
        shutil.copy2(source, package / source.name)
    for name in [
        "baseline.py",
        "hybrid.py",
        "consensus.py",
        "requirements.txt",
        "requirements-ml.txt",
    ]:
        shutil.copy2(name, output / name)
    (output / "RUN.md").write_text(
        "Run with this directory on PYTHONPATH: python -m casmi_ml.secondary_inference --recipe deployment_recipe.json --data-dir /path/to/data --coconut /path/to/coconut_structures.parquet --output submission.csv\nNo network call or upload is performed by inference. MetFrag requires Java.\n"
    )


def prepare_release(root=ROOT, output=None, jar=None):
    root = Path(root)
    output = Path(output or root / "release")
    acceptance = json.loads((root / "chemical_acceptance.json").read_text())
    if not acceptance["accepted"]:
        raise ValueError("Independent chemistry acceptance did not pass")
    selected = root / "chemical_selection.json"
    if digest(selected) != acceptance["selection_sha256"]:
        raise ValueError("Selection changed after acceptance")
    winner = json.loads(selected.read_text())["winner"]
    recipe = json.loads(
        Path("artifacts/scale_20260929/deployment_recipe.json").read_text()
    )
    output.mkdir(parents=True, exist_ok=True)
    source = Path(recipe["checkpoint"])
    if digest(source) != recipe["checkpoint_sha256"]:
        raise ValueError("Baseline encoder checksum mismatch")
    if not (output / "model.pt").exists():
        shutil.copy2(source, output / "model.pt")
    if digest(output / "model.pt") != recipe["checkpoint_sha256"]:
        raise ValueError("Packaged encoder checksum mismatch")
    recipe["checkpoint"] = "model.pt"
    recipe["selection"] = "chemical_selection.json"
    recipe["holdout_report"] = "chemical_acceptance.json"
    recipe["chemistry"] = {
        "rules_version": VERSION,
        "weight": winner["weight"],
        "component": winner["component"],
    }
    if winner["component"] in ["fragment", "combined_fragment"]:
        if jar is None:
            raise ValueError("Selected fragmentation recipe requires --metfrag-jar")
        jar = Path(jar)
        protocol = json.loads((root / "chemical_protocol.json").read_text())
        if digest(jar) != protocol["fragmenter_sha256"]:
            raise ValueError("Fragmenter mismatch")
        if not (output / "metfrag.jar").exists():
            shutil.copy2(jar, output / "metfrag.jar")
        if digest(output / "metfrag.jar") != digest(jar):
            raise ValueError("Packaged fragmenter checksum mismatch")
        for attribution in jar.parent.iterdir():
            if attribution.is_file() and attribution.suffix.lower() in [
                ".md",
                ".txt",
                ".json",
            ]:
                target = output / "metfrag_attribution" / attribution.name
                target.parent.mkdir(exist_ok=True)
                shutil.copy2(attribution, target)
        recipe["chemistry"].update({"jar": "metfrag.jar", "jar_sha256": digest(jar)})
    copy_inference_source(output)
    recipe.update(
        {
            "status": "passed_chemistry_independent_cpu_acceptance",
            "kaggle_submitted": False,
        }
    )
    freeze(output / "deployment_recipe.json", recipe)
    for name in [
        "protocol.json",
        "chemical_protocol.json",
        "chemical_selection.json",
        "chemical_acceptance.json",
    ]:
        shutil.copy2(root / name, output / name)
    write_json(
        output / "manifest.json",
        {
            "files": {
                str(p.relative_to(output)): digest(p)
                for p in output.rglob("*")
                if p.is_file() and p.name != "manifest.json"
            },
            "uploaded": False,
            "external_tools_require_competition_rule_check": True,
        },
    )
    return output / "deployment_recipe.json"


def prepare_representation_release(root=ROOT, output=None):
    root = Path(root)
    output = Path(output or root / "representation_release")
    acceptance = json.loads((root / "representation_acceptance.json").read_text())
    if not acceptance["accepted"]:
        raise ValueError("Independent representation acceptance did not pass")
    selection_path = root / "representation_selection.json"
    if digest(selection_path) != acceptance["selection_sha256"]:
        raise ValueError("Representation selection changed")
    winner = json.loads(selection_path.read_text())["winner"]
    if digest(winner["checkpoint"]) != winner["sha256"]:
        raise ValueError("Representation checkpoint changed")
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(winner["checkpoint"], output / "model.pt")
    recipe = json.loads(
        Path("artifacts/scale_20260929/deployment_recipe.json").read_text()
    )
    recipe.update(
        {
            "encoder_family": "research_peak",
            "checkpoint": "model.pt",
            "checkpoint_sha256": winner["sha256"],
            "kaggle_submitted": False,
            "status": "passed_representation_independent_cpu_acceptance",
        }
    )
    copy_inference_source(output)
    freeze(output / "deployment_recipe.json", recipe)
    return output / "deployment_recipe.json"


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--representation", action="store_true")
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path)
    p.add_argument("--metfrag-jar", type=Path)
    a = p.parse_args()
    print(
        prepare_representation_release(a.root, a.output)
        if a.representation
        else prepare_release(a.root, a.output, a.metfrag_jar)
    )


if __name__ == "__main__":
    main()
