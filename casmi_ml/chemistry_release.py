"""Prepare the accepted chemistry model for offline Kaggle code submission."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT


def prepare(root=ROOT, output=Path("kaggle_release_chemistry")):
    root, output = Path(root), Path(output)
    accepted = json.loads((root / "chemical_acceptance.json").read_text())
    if not accepted["accepted"]:
        raise ValueError("Independent chemistry acceptance required")
    original = root / "release"
    manifest = json.loads((original / "manifest.json").read_text())["files"]
    if not all(digest(original / p) == h for p, h in manifest.items()):
        raise ValueError("Local release manifest mismatch")
    bundle = output / "bundle"
    bundle.mkdir(parents=True, exist_ok=True)
    for name in manifest:
        target = bundle / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(original / name, target)
    # Execution budget guard only; accepted rules and weights remain unchanged.
    for name in ["metfrag.py", "secondary_inference.py"]:
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    recipe = json.loads((bundle / "deployment_recipe.json").read_text())
    recipe["chemistry"]["fragment_seconds"] = 1200
    write_json(bundle / "deployment_recipe.json", recipe)
    wheels = bundle / "wheels"
    wheels.mkdir(exist_ok=True)
    for wheel in Path("kaggle_release_scale/bundle/wheels").glob("rdkit*.whl"):
        shutil.copy2(wheel, wheels / wheel.name)
    if len(list(wheels.glob("rdkit*.whl"))) != 1:
        raise ValueError("Exactly one validated RDKit wheel required")
    shutil.copy2("kaggle_release_scale/bundle/runtime_versions.json", bundle)
    java = Path("external/metfrag/java21")
    runtime = json.loads((java / "manifest.json").read_text())
    archive = java / runtime["package"]["name"]
    if digest(archive) != runtime["sha256"]:
        raise ValueError("Java runtime checksum mismatch")
    shutil.copy2(archive, bundle / "java-runtime.bin")
    write_json(bundle / "java-runtime.json", runtime)
    shutil.copy2(root / "release_verification.json", bundle)
    shutil.copy2(root / "REPORT.md", bundle)
    sums = {
        str(p.relative_to(bundle)): digest(p)
        for p in bundle.rglob("*")
        if p.is_file()
        and p.name not in ["manifest.json", "SHA256SUMS.json"]
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    write_json(bundle / "SHA256SUMS.json", sums)
    dataset = output / "dataset"
    dataset.mkdir(exist_ok=True)
    with zipfile.ZipFile(
        dataset / "casmi_chemistry_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as zip_file:
        for name in [*sums, "SHA256SUMS.json"]:
            zip_file.write(bundle / name, name)
    write_json(
        dataset / "dataset-metadata.json",
        {
            "title": "CASMI 2026 Validated MetFrag Chemistry Bundle",
            "id": "giaok246/casmi-2026-metfrag-chemistry-bundle",
            "licenses": [{"name": "other"}],
            "description": "Frozen 60K fingerprint encoder and independently accepted offline MetFrag 2.6.11 reranking; upstream tool attribution and checksums included. Private competition inference assets.",
        },
    )
    notebook = output / "notebook"
    notebook.mkdir(exist_ok=True)
    code = '''from pathlib import Path
import hashlib, importlib.metadata, json, os, shutil, subprocess, sys, tarfile, zipfile
archives = list(Path('/kaggle/input').rglob('casmi_chemistry_bundle.zip'))
bundle = Path('/kaggle/temp/casmi_chemistry_runtime')
bundle.mkdir(parents=True, exist_ok=True)
if len(archives) == 1:
    with zipfile.ZipFile(archives[0]) as archive:
        archive.extractall(bundle)
else:
    roots = list(Path('/kaggle/input').rglob('deployment_recipe.json'))
    assert len(roots) == 1, roots
    shutil.copytree(roots[0].parent, bundle, dirs_exist_ok=True)
for name, expected in json.loads((bundle / 'SHA256SUMS.json').read_text()).items():
    assert hashlib.sha256((bundle / name).read_bytes()).hexdigest() == expected, name
required = json.loads((bundle / 'runtime_versions.json').read_text())
try:
    installed_rdkit = importlib.metadata.version('rdkit')
except importlib.metadata.PackageNotFoundError:
    installed_rdkit = None
if installed_rdkit != required['rdkit']:
    wheels = list((bundle / 'wheels').glob('rdkit*.whl'))
    assert len(wheels) == 1
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--no-index', '--no-deps', str(wheels[0])])
java_root = bundle / 'java'
java_root.mkdir(exist_ok=True)
with tarfile.open(bundle / 'java-runtime.bin') as archive:
    archive.extractall(java_root, filter='data')
java = list(java_root.glob('*/bin/java'))
assert len(java) == 1
java[0].chmod(0o755)
os.environ['PATH'] = str(java[0].parent) + os.pathsep + os.environ['PATH']
os.environ['PYTHONPATH'] = str(bundle)
subprocess.check_call(['java', '-version'])
smoke = """from casmi_ml.metfrag import MetFrag
from casmi_ml.chemistry import composition_mass
row = {'precursor_mz': composition_mass('C2H6O') + 1.007276466621,
       'adduct': '[M+H]+', 'instrument_type': 'QTOF',
       'ms2_mzs': [31.01839, 29.038576], 'ms2_normalized_intensities': [1.0, 0.6]}
result = MetFrag('metfrag.jar', 'smoke-cache').score(row, {'ethanol': 'CCO', 'ether': 'COC'})
assert result['status'] == 'complete', result
assert set(result['scores']) == {'ethanol', 'ether'}, result
print('MetFrag smoke:', result)
"""
subprocess.check_call([sys.executable, '-c', smoke], cwd=bundle)
subprocess.check_call([sys.executable, '-m', 'casmi_ml.secondary_inference',
    '--recipe', str(bundle / 'deployment_recipe.json'),
    '--data-dir', '/kaggle/input/competition',
    '--coconut', '/kaggle/input/coconut_structures.parquet',
    '--output', '/kaggle/working/submission.csv'], cwd=bundle)
report = json.loads(Path('/kaggle/working/submission.csv.report.json').read_text())
assert report['seconds'] < 1800, report
assert report['peak_rss_mib'] < 8192, report
print(report)
'''
    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            }
        },
        "cells": [
            {
                "id": "overview",
                "cell_type": "markdown",
                "metadata": {},
                "source": "# CASMI 2026 Validated MetFrag Chemistry\nFrozen 60K encoder, independent chemistry acceptance, protected historical retrieval. CPU-only offline inference, fragmentation budget fallback.\n",
            },
            {
                "id": "inference",
                "cell_type": "code",
                "metadata": {},
                "source": code,
                "outputs": [],
                "execution_count": None,
            },
        ],
    }
    write_json(notebook / "casmi_chemistry.ipynb", nb)
    write_json(
        notebook / "kernel-metadata.json",
        {
            "id": "giaok246/casmi-2026-metfrag-chemistry-inference",
            "title": "CASMI 2026 MetFrag Chemistry Inference",
            "code_file": "casmi_chemistry.ipynb",
            "language": "python",
            "kernel_type": "notebook",
            "is_private": True,
            "enable_gpu": False,
            "enable_tpu": False,
            "enable_internet": False,
            "competition_sources": ["enveda-CASMI26-molecule-id-mass-spectra"],
            "dataset_sources": [
                "giaok246/casmi-2026-metfrag-chemistry-bundle",
                "aidensong123/casmi26-coconut-202609",
            ],
            "kernel_sources": [],
            "model_sources": [],
        },
    )
    write_json(
        output / "status.json",
        {
            "status": "prepared_locally",
            "uploaded": False,
            "submitted": False,
            "local_holdout_accepted": True,
            "fragment_seconds": 1200,
        },
    )
    return output


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--output", type=Path, default=Path("kaggle_release_chemistry"))
    args = p.parse_args()
    print(prepare(args.root, args.output))


if __name__ == "__main__":
    main()
