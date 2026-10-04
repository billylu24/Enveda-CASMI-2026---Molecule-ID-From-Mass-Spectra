"""Independent per-molecule original generation on two GPUs, one shared deadline."""

import argparse
import inspect
import json
import os
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd
import torch

from casmi_ml.data import write_json
from casmi_ml.generation_inference import predict as original_predict
from casmi_ml.inference import validate_submission
from casmi_ml.metfrag import digest

PATH_FIELDS = {
    "checkpoint",
    "test_path",
    "baseline_csv",
    "output",
    "routing_csv",
    "encoder_path",
    "full_rankings",
    "critic_checkpoint",
    "output_full_rankings",
}


def partitions(frame, workers=2):
    if workers != 2 or "molecule_id" not in frame or frame.molecule_id.isna().any():
        raise ValueError("Two whole-molecule partitions with valid opaque IDs required")
    keys = list(frame.molecule_id.unique())
    if len(keys) < 2:
        raise ValueError("At least two molecules required for partition verification")
    return [keys[i::workers] for i in range(workers)]


def ordered_merge(parts, order, identifier="molecule_id"):
    merged = pd.concat(parts, ignore_index=True)
    if merged[identifier].duplicated().any() or set(merged[identifier]) != set(order):
        raise ValueError("Complete disjoint original molecular output required")
    return merged.set_index(identifier).loc[order].reset_index()


def run_worker(config):
    payload = json.loads(Path(config).read_text())
    arguments = payload["arguments"]
    remaining = payload["deadline"] - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Original shared generation deadline expired before worker")
    arguments["seconds"] = remaining
    original_predict(**arguments)


def parallel_predict(*args, force_partition=False, **kwargs):
    # On a one-GPU deployment preserve the original path. Verification may force
    # two sequential partitions on that GPU to check identity without contention.
    gpu_count = torch.cuda.device_count()
    if gpu_count < 2 and not force_partition:
        return original_predict(*args, **kwargs)
    if gpu_count < 1:
        raise RuntimeError("GPU required for partitioned generator verification")
    bound = inspect.signature(original_predict).bind(*args, **kwargs)
    bound.apply_defaults()
    parameters = dict(bound.arguments)
    for name in PATH_FIELDS:
        if parameters.get(name) is not None:
            parameters[name] = str(Path(parameters[name]).resolve())
    started = time.monotonic()
    deadline = started + parameters["seconds"]
    frame = pd.read_parquet(parameters["test_path"])
    order = list(frame.molecule_id.unique())
    shards = partitions(frame)
    baseline = pd.read_csv(parameters["baseline_csv"])
    routing = pd.read_csv(
        parameters["routing_csv"] or parameters["baseline_csv"] + ".routing.csv"
    )
    full = (
        json.loads(Path(parameters["full_rankings"]).read_text())
        if parameters["full_rankings"]
        else None
    )
    reports, outputs, audits, fulls = [], [], [], []
    with tempfile.TemporaryDirectory(prefix="casmi-dual-generation-") as root:
        children = []
        destinations = []
        try:
            for i, keys in enumerate(shards):
                dest = Path(root) / str(i)
                dest.mkdir()
                destinations.append(dest)
                selected = set(keys)
                frame[frame.molecule_id.isin(selected)].to_parquet(
                    dest / "test.parquet", index=False
                )
                baseline[baseline.molecule_id.isin(selected)].to_csv(
                    dest / "baseline.csv", index=False
                )
                routing[routing.molecule_id.isin(selected)].to_csv(
                    dest / "routing.csv", index=False
                )
                arguments = dict(parameters)
                arguments.update(
                    test_path=str(dest / "test.parquet"),
                    baseline_csv=str(dest / "baseline.csv"),
                    routing_csv=str(dest / "routing.csv"),
                    output=str(dest / "generated.csv"),
                )
                if full is not None:
                    write_json(
                        dest / "baseline_full.json",
                        [r for r in full if r["molecule_id"] in selected],
                    )
                    arguments["full_rankings"] = str(dest / "baseline_full.json")
                arguments["output_full_rankings"] = str(dest / "full.json")
                write_json(
                    dest / "worker.json", {"arguments": arguments, "deadline": deadline}
                )
                env = dict(
                    os.environ, CUDA_VISIBLE_DEVICES=str(i if gpu_count >= 2 else 0)
                )
                stream = (dest / "execution.log").open("w")
                child = subprocess.Popen(
                    [
                        sys.executable,
                        "-m",
                        "casmi_ml.parallel_generation_runtime",
                        "--worker",
                        str(dest / "worker.json"),
                    ],
                    env=env,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                )
                children.append((child, stream))
                if gpu_count < 2:
                    child.wait(timeout=max(0.001, deadline - time.monotonic()))
                    if child.returncode:
                        raise RuntimeError((dest / "execution.log").read_text()[-3000:])
            for child, _ in children:
                child.wait(timeout=max(0.001, deadline - time.monotonic()))
                if child.returncode:
                    raise RuntimeError("Original generation worker failed")
            for dest in destinations:
                report = json.loads((dest / "generated.csv.report.json").read_text())
                audit = pd.read_csv(dest / "generated.csv.generation.csv")
                if audit.status.str.contains("budget").any():
                    raise ValueError(
                        "Parallel generation cannot conceal budget fallbacks"
                    )
                reports.append(report)
                audits.append(audit)
                outputs.append(pd.read_csv(dest / "generated.csv"))
                fulls.extend(json.loads((dest / "full.json").read_text()))
        finally:
            for child, stream in children:
                if child.poll() is None:
                    child.kill()
                child.wait()
                stream.close()
    result = ordered_merge(outputs, order)
    validate_submission(frame, result)
    result.to_csv(parameters["output"], index=False)
    audit = ordered_merge(audits, order)
    audit.to_csv(parameters["output"] + ".generation.csv", index=False)
    records = {r["molecule_id"]: r for r in fulls}
    if len(records) != len(fulls) or set(records) != set(order):
        raise ValueError("Complete disjoint original full generation rankings required")
    if parameters["output_full_rankings"]:
        write_json(parameters["output_full_rankings"], [records[k] for k in order])
    bindings = [
        {
            k: v
            for k, v in r.items()
            if k not in {"seconds", "molecules", "peak_rss_mib", "baseline_sha256"}
        }
        for r in reports
    ]
    if bindings[0] != bindings[1]:
        raise ValueError("Generator model/rules differ across partitions")
    report = dict(reports[0])
    report.update(
        molecules=len(order),
        seconds=time.monotonic() - started,
        baseline_sha256=digest(parameters["baseline_csv"]),
        peak_rss_mib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
    )
    report["parallel_generation_runtime"] = {
        "workers": 2,
        "concurrent": gpu_count >= 2,
        "molecules_per_partition": [len(p) for p in shards],
        "shared_original_deadline": True,
        "source_sha256": digest(__file__),
        "original_generation_source_sha256": digest("casmi_ml/generation_inference.py"),
        "worker_peak_rss_mib": [r["peak_rss_mib"] for r in reports],
        "scope": "Original single-molecule condition/decoder/sample count/independent spectrum seed/critic/insertion unchanged. Whole molecules partitioned, outputs restored to original order. OneGPU forced partitions sequential for identity verification; twoGPU partitions concurrent. No query predictions packaged and original global physical deadline unchanged.",
    }
    write_json(parameters["output"] + ".report.json", report)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True)
    run_worker(parser.parse_args().worker)


if __name__ == "__main__":
    main()
