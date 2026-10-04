"""Same streaming requests with one CPU scoring thread and per-call catalog reuse."""

import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from casmi_ml import chembl_routed_inference as base
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.streaming_fragment_prefetch import extend_streamed


def extend_balanced(*args, **kwargs):
    root = Path(args[5] if len(args) > 5 else kwargs["root"])
    config = args[6] if len(args) > 6 else kwargs["config"]
    catalog_path = (root / config["catalog"]).resolve()
    output = Path(args[4] if len(args) > 4 else kwargs["output"])
    original_read, constructor, configure = (
        base.pd.read_parquet,
        base.CandidateIndex,
        base.configure,
    )
    original_threads = torch.get_num_threads()
    catalog = index = None
    timings = {
        "catalog_reads": 0,
        "catalog_constructions": 0,
        "catalog_read_seconds": 0.0,
        "catalog_construct_seconds": 0.0,
    }

    def read(path, *read_args, **read_kwargs):
        nonlocal catalog
        if not isinstance(path, (str, Path)) or Path(path).resolve() != catalog_path:
            return original_read(path, *read_args, **read_kwargs)
        if read_args or read_kwargs:
            raise ValueError("Exact original whole external catalog read required")
        if catalog is None:
            started = time.monotonic()
            catalog = original_read(path)
            timings["catalog_read_seconds"] += time.monotonic() - started
            timings["catalog_reads"] += 1
        return catalog

    def shared_index(frame):
        nonlocal index
        if frame is not catalog:
            raise ValueError("Only the unchanged full catalog may reuse its index")
        if index is None:
            started = time.monotonic()
            index = constructor(frame)
            timings["catalog_construct_seconds"] += time.monotonic() - started
            timings["catalog_constructions"] += 1
        return index

    def configure_cpu(seed=42, threads=4):
        # Original seed and original scoring graph; only CPU concurrency changes.
        return configure(seed=seed, threads=1)

    try:
        with (
            patch.object(base.pd, "read_parquet", new=read),
            patch.object(base, "CandidateIndex", new=shared_index),
            patch.object(base, "configure", new=configure_cpu),
        ):
            result = extend_streamed(*args, **kwargs)
    finally:
        torch.set_num_threads(original_threads)
    report_path = Path(str(output) + ".external.report.json")
    report = json.loads(report_path.read_text())
    report["balanced_cpu_runtime"] = {
        "source_sha256": digest(__file__),
        "cpu_scoring_threads": 1,
        "physical_workers": 2,
        "candidate_threads_per_worker": 2,
        "original_deadlines_retained": True,
        **timings,
        "scope": "Same original scoring graph/seeds/candidate order and physical requests; reduce competing CPU scoring thread count and reuse one exact catalog/index across planning and consumption within this call. No packaged query evidence, no resource/accuracy acceptance from configuration alone.",
    }
    write_json(report_path, report)
    return result
