"""Freeze a source-qualified parallel runtime amendment of the same236 science."""

import argparse
import json
import shutil
import zipfile
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.portable_packed_release import prepare as packed_prepare
from casmi_ml.research_loop import release_identity
from casmi_ml.verified_best_release import VARIANT

CONTROL = Path("artifacts/research_loop/rounds/0354_fragment_prefetch_zero_control")


def prepare(old, output):
    control = json.loads((CONTROL / "report.json").read_text())
    if (
        not control["control_passed"]
        or control["development_budget_fallbacks"]
        or control["incremental_new_failure_contexts"]
        or control["prefetch_source_sha256"]
        != digest("casmi_ml/portable_fragment_prefetch.py")
    ):
        raise ValueError("Current source all3340 exact parallel control required")
    output = Path(output)
    packed_prepare(
        old,
        output,
        dataset_id="giaok246/casmi26-236-prefetch-best-bundle",
        kernel_id="giaok246/casmi26-236-prefetch-verified-best",
    )
    bundle = output / "bundle"
    for name in ("portable_fragment_prefetch.py", "portable_prefetch_pipeline.py"):
        shutil.copy2(Path("casmi_ml") / name, bundle / "casmi_ml" / name)
    evidence = json.loads((bundle / "release_evidence.json").read_text())
    evidence["portable_fragment_prefetch"] = {
        "source_sha256": digest("casmi_ml/portable_fragment_prefetch.py"),
        "control_report_sha256": digest(CONTROL / "report.json"),
        "full_rank_matches": {"unknown": 2000, "known": 1340},
        "physical_workers": 2,
        "scope": "Only original request scheduling and same-invocation critic/probability memoization. Original models/science/request dictionaries/thread2/deadlines unchanged. No precomputed query-dependent fragments or predictions included. New cold400+fresh75+platform400 resource and exact-rank checks required.",
    }
    write_json(bundle / "release_evidence.json", evidence)
    notebook_path = output / "notebook/casmi_chemistry.ipynb"
    notebook = json.loads(notebook_path.read_text())
    notebook["cells"][1]["source"] = notebook["cells"][1]["source"].replace(
        "casmi_ml.portable_packed_pipeline", "casmi_ml.portable_prefetch_pipeline"
    )
    write_json(notebook_path, notebook)
    sums = {
        str(p.relative_to(bundle)): digest(p)
        for p in bundle.rglob("*")
        if p.is_file()
        and p.name not in ("manifest.json", "SHA256SUMS.json")
        and "__pycache__" not in p.parts
        and p.suffix != ".pyc"
    }
    write_json(bundle / "SHA256SUMS.json", sums)
    with zipfile.ZipFile(
        output / "dataset/casmi_chemistry_bundle.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for name in [*sums, "SHA256SUMS.json"]:
            archive.write(bundle / name, name)
    status = json.loads((output / "status.json").read_text())
    status["identity"] = release_identity(output, sums, VARIANT)
    write_json(output / "status.json", status)
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    print(prepare(**vars(parser.parse_args())))


if __name__ == "__main__":
    main()
