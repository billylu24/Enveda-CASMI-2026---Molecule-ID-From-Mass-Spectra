"""Build the self-contained, offline Kaggle historical-hybrid chemistry notebook."""

import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
RELEASE = Path(__file__).resolve().parent
SOURCES = (
    "baseline.py",
    "hybrid.py",
    "casmi_ml/__init__.py",
    "casmi_ml/chemical_priors.py",
    "casmi_ml/hybrid_chemistry.py",
)
COCONUT_SHA256 = "6d8bd9206fa576fecd2741c020ba64f5f4e60609f767bd8aad8a6628a5b87bbd"


def code_archive():
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name in SOURCES:
            source = (ROOT / name).read_bytes()
            compile(source, name, "exec")
            entry = zipfile.ZipInfo(name, date_time=(2026, 10, 2, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            archive.writestr(entry, source)
    return buffer.getvalue()


def cell(kind, source):
    result = {"cell_type": kind, "id": hashlib.sha256(source.encode()).hexdigest()[:12],
              "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def build():
    archive = code_archive()
    encoded = base64.b64encode(archive).decode("ascii")
    archive_sha = hashlib.sha256(archive).hexdigest()
    sums_sha = hashlib.sha256((RELEASE / "bundle/SHA256SUMS.json").read_bytes()).hexdigest()
    bootstrap = '''from pathlib import Path
import base64, hashlib, io, json, subprocess, sys, zipfile

INPUT = Path("/kaggle/input")
WORKING = Path("/kaggle/working")
WORKING.mkdir(parents=True, exist_ok=True)

def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()

def unique(paths, description):
    paths = list(paths)
    if len(paths) != 1:
        raise FileNotFoundError(f"Expected one {description}, found {paths}")
    return paths[0]

def extract_safely(archive, destination):
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for member in archive.infolist():
        target = (destination / member.filename).resolve()
        if not target.is_relative_to(destination):
            raise ValueError(f"Unsafe archive member: {member.filename}")
    archive.extractall(destination)

# Kaggle may expose dataset zip contents directly or mount the original zip.
mounted_catalogs = list(INPUT.rglob("chemical_structures.parquet"))
if mounted_catalogs:
    bundle = unique(mounted_catalogs, "public chemical catalog").parent
else:
    input_zip = unique(INPUT.rglob("casmi_chemistry_inputs.zip"), "chemistry input archive")
    extracted = WORKING / "casmi_chemistry_inputs"
    with zipfile.ZipFile(input_zip) as source:
        extract_safely(source, extracted)
    bundle = unique(extracted.rglob("chemical_structures.parquet"), "extracted public catalog").parent

sums_path = bundle / "SHA256SUMS.json"
if sha256(sums_path) != SUMS_SHA256:
    raise ValueError("Chemistry input checksum manifest differs from the frozen package")
checksums = json.loads(sums_path.read_text())
for relative, expected in checksums.items():
    path = (bundle / relative).resolve()
    if not path.is_relative_to(bundle.resolve()):
        raise ValueError(f"Unsafe checksum path: {relative}")
    if not path.is_file() or sha256(path) != expected:
        raise ValueError(f"Chemistry input checksum mismatch: {relative}")
print(f"Verified {len(checksums)} input files from {bundle}")

# Install the checksum-verified wheel without consulting package indexes.
if sys.version_info[:2] != (3, 12):
    raise RuntimeError(f"The packaged CPython 3.12 wheel requires Python 3.12; got {sys.version}")
wheel = unique((bundle / "wheels").glob("rdkit-2026.3.3-cp312-cp312-manylinux*.whl"), "pinned RDKit wheel")
subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                "--force-reinstall", str(wheel)], check=True)
import rdkit
if rdkit.__version__ != "2026.03.3":
    raise RuntimeError(f"Expected RDKit 2026.03.3, found {rdkit.__version__}")
print("RDKit", rdkit.__version__)

source_bytes = base64.b64decode(CODE_BASE64, validate=True)
if hashlib.sha256(source_bytes).hexdigest() != CODE_SHA256:
    raise ValueError("Embedded source archive checksum mismatch")
code_dir = WORKING / "chemistry_code"
with zipfile.ZipFile(io.BytesIO(source_bytes)) as source:
    extract_safely(source, code_dir)
sys.path.insert(0, str(code_dir))

CATALOG_PATH = bundle / "chemical_structures.parquet"
DICTIONARY_PATH = bundle / "chemical_priors.json"
COCONUT_PATH = unique(INPUT.rglob("coconut_structures.parquet"), "COCONUT structure parquet")
if sha256(COCONUT_PATH) != COCONUT_SHA256:
    raise ValueError("COCONUT checksum mismatch; attach aidensong123/casmi26-coconut-202609")
TRAIN_PATH = unique(INPUT.rglob("train.parquet"), "competition train.parquet")
TEST_PATH = unique(INPUT.rglob("test.parquet"), "competition test.parquet")
if TRAIN_PATH.parent != TEST_PATH.parent:
    raise ValueError("Competition train/test files must share the mounted data directory")
DATA_DIR = TRAIN_PATH.parent
print(json.dumps({"competition": str(DATA_DIR), "coconut": str(COCONUT_PATH),
                  "public_catalog": str(CATALOG_PATH), "dictionary": str(DICTIONARY_PATH),
                  "embedded_source_sha256": CODE_SHA256}, indent=2))
from casmi_ml.hybrid_chemistry import predict, behavior_check
'''
    constants = (
        f"CODE_BASE64 = {encoded!r}\n"
        f"CODE_SHA256 = {archive_sha!r}\n"
        f"SUMS_SHA256 = {sums_sha!r}\n"
        f"COCONUT_SHA256 = {COCONUT_SHA256!r}\n\n"
    )
    prediction = '''submission, report = predict(
    DATA_DIR, COCONUT_PATH, CATALOG_PATH, DICTIONARY_PATH,
    output=WORKING / "submission.csv", chemistry_weight=0.1,
    catalog_enabled=True, rules_enabled=True,
)
print(json.dumps(report, ensure_ascii=False, indent=2))
print(f"submission.csv: {len(submission)} molecules; candidate and SMILES checks passed")
display(submission.head())
'''
    behavior = '''check = behavior_check(
    TRAIN_PATH, COCONUT_PATH, CATALOG_PATH, DICTIONARY_PATH,
    output=WORKING / "behavior_check.json", chemistry_weight=0.1, count=32,
)
four_arms = {
    name: {key: value for key, value in group.items()
           if key not in {"audit", "identity_results"}}
    for name, group in check["groups"].items()
}
print("Behavior checks only: these previously observed molecules are not independent validation.")
print("The weight is predetermined and is not selected from these four-arm results.")
print(json.dumps({"status": check["status"], "four_arms": four_arms}, indent=2))
'''
    explanation = """# CASMI historical hybrid with public structures and chemical evidence

This experimental notebook extends the historical `hybrid.py` implementation associated with the previous public score **0.176**. It adds the ChEBI/LMSD public catalog and five project-authored chemical motif rules. A library confidence of **0.5** protects historical rankings; the chemical weight **0.1** is predetermined. Neither parameter was selected using the behavior check below. This new version has no measured leaderboard improvement.

The GAN has **not been trained**. This runtime uses the historical spectrum-library/COCONUT analog method and requires no neural checkpoint. The five cited soft rules are project-authored; MS-FINDER dictionary resource files with unverified data-specific licensing are **not distributed** in this input package.

Attach the competition data, `xiaoyuzhoux120/casmi-2026-chebi-lmsd-chemistry-inputs`, and `aidensong123/casmi26-coconut-202609`. Internet access is unnecessary: source code is embedded, inputs are checksum verified, and the packaged RDKit wheel is installed offline. The output for submission is `/kaggle/working/submission.csv`.

The final cell checks 32 previously observed natural-product examples after excluding their reference keys. Arms A/B/C/D switch the public catalog and chemical rules independently. These are behavior checks, **not independent validation** or evidence of generalization. `behavior_check.json` preserves the detailed results. Attribution and license records accompany the public input dataset.
"""
    notebook = {
        "cells": [cell("markdown", explanation), cell("code", constants + bootstrap),
                  cell("code", prediction), cell("code", behavior)],
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                     "language_info": {"name": "python", "version": "3.12"},
                     "kaggle": {"accelerator": "none", "isInternetEnabled": False}},
        "nbformat": 4, "nbformat_minor": 5,
    }
    for index, entry in enumerate(notebook["cells"]):
        if entry["cell_type"] == "code":
            compile("".join(entry["source"]), f"chemical_priors_hybrid.ipynb:cell{index}", "exec")
    output = RELEASE / "notebook/chemical_priors_hybrid.ipynb"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps({"notebook": str(output), "bytes": output.stat().st_size,
                      "code_cells_compiled": 3, "embedded_code_sha256": archive_sha,
                      "input_manifest_sha256": sums_sha}, indent=2))
    return output


if __name__ == "__main__":
    build()
