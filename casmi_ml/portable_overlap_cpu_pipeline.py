"""Experimental local overlap with original generator CPU worker concurrency4->1."""

import argparse
import json
import subprocess
from pathlib import Path
from unittest.mock import patch

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.portable_verified_overlap_pipeline import run as original_run


def redirect_worker(command):
    if (
        isinstance(command, (list, tuple))
        and len(command) >= 3
        and list(command[1:3]) == ["-m", "casmi_ml.generation_handoff_stream"]
    ):
        command = list(command)
        command[2] = "casmi_ml.generation_handoff_cpu_worker"
    return command


def run(recipe, data_dir, coconut, output, force_partitions=False):
    popen = subprocess.Popen
    workers = []

    def launch(command, *args, **kwargs):
        changed = redirect_worker(command)
        if changed != command:
            workers.append("original_generation_cpu_threads1")
        return popen(changed, *args, **kwargs)

    with patch.object(subprocess, "Popen", new=launch):
        result = original_run(recipe, data_dir, coconut, output, force_partitions)
    report_path = Path(str(output) + ".report.json")
    report = json.loads(report_path.read_text())
    report["generation_cpu_runtime"] = {
        "workers": len(workers),
        "torch_cpu_threads": 1,
        "worker_source_sha256": digest("casmi_ml/generation_handoff_cpu_worker.py"),
        "pipeline_source_sha256": digest(__file__),
        "scope": "Experimental local only. Original per-molecule seeded generator "
        "and AST/128samples/model/CUDA work/critic/gates unchanged except worker "
        "Torch CPU threads4->1. No BLAS environment or twoJVM/two candidate-thread "
        "physical settings/deadline change. Require all400 generation CSV/audit/"
        "handoffs/events and full400Top25/265rawdict exact, fresh75/full3340/"
        "frozen-source/resource/platform proofs before any release amendment.",
    }
    write_json(report_path, report)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--force-partitions", action="store_true")
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
