"""Observe unchanged original overlapping requests, including failures, without tuning."""

import json
import os
import resource
import threading
import time
from pathlib import Path
from unittest.mock import patch

from casmi_ml import parallel_generation_runtime as generation
from casmi_ml import portable_fragment_prefetch as physical
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.portable_verified_overlap_pipeline import run as original_run


def measure(recipe, data, coconut, output, summary, *, force_partitions=False):
    summary = Path(summary)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    lock = threading.Lock()
    events, discoveries, phases = [], {}, []
    original_record, original_perform = (
        physical.FragmentPrefetch.record,
        physical.FragmentPrefetch.perform,
    )
    original_generate = generation.parallel_predict
    success, failure, result = False, None, None

    def record(pool, engine, cache_path, candidates, peaks, mass, adduct):
        key = str(cache_path)
        with lock:
            discoveries.setdefault(key, time.monotonic() - started)
            write_json(
                summary.parent / "private_request_inputs" / (Path(key).stem + ".json"),
                {
                    "cache_key": Path(key).stem,
                    "candidates": candidates,
                    "peaks": peaks,
                    "mass": mass,
                    "adduct": adduct,
                    "scope": "Original invocation request only; private, never packaged or published.",
                },
            )
        return original_record(
            pool, engine, cache_path, candidates, peaks, mass, adduct
        )

    def perform(pool, request):
        begun = time.monotonic()
        key = str(request[0])
        with lock:
            event = {
                "request_digest": Path(key).stem,
                "candidates": len(request[1]),
                "peaks": len(request[2]),
                "discovered_seconds": discoveries.get(key),
                "started_seconds": begun - started,
                "global_remaining_seconds": None
                if pool.deadline is None
                else pool.deadline - begun,
                "configured_request_timeout": pool.parameters["timeout"],
                "candidate_threads": pool.parameters["threads"],
                "status": "running",
            }
            events.append(event)
            write_json(summary.with_name("request_events.json"), events)
        try:
            value = original_perform(pool, request)
            event["status"] = value.get("status")
            return value
        except BaseException as error:
            event["status"] = "exception"
            event["error"] = type(error).__name__ + ": " + str(error)
            raise
        finally:
            with lock:
                event["seconds"] = time.monotonic() - begun
                engine = getattr(pool.local, "engine", None)
                if engine is not None:
                    event["worker_requests"] = engine.runner.requests
                write_json(summary.with_name("request_events.json"), events)
                if len(events) % 10 == 0 or event["status"] != "complete":
                    print("ORIGINAL_PHYSICAL_EVENT", json.dumps(event), flush=True)

    def generate(*args, **kwargs):
        phase = {"stage": "generation", "started_seconds": time.monotonic() - started}
        phases.append(phase)
        try:
            return original_generate(*args, **kwargs)
        finally:
            phase["ended_seconds"] = time.monotonic() - started
            print("ORIGINAL_GENERATION_PHASE", json.dumps(phase), flush=True)

    try:
        with (
            patch.object(physical.FragmentPrefetch, "record", new=record),
            patch.object(physical.FragmentPrefetch, "perform", new=perform),
            patch.object(generation, "parallel_predict", new=generate),
        ):
            result = original_run(recipe, data, coconut, output, force_partitions)
        success = True
        return result
    except BaseException as error:
        failure = type(error).__name__ + ": " + str(error)
        raise
    finally:
        write_json(
            summary,
            {
                "diagnostic_only": True,
                "competition_submission": False,
                "success": success,
                "error": failure,
                "source_sha256": digest(__file__),
                "wall_seconds": time.monotonic() - started,
                "parent_peak_rss_mib": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                / 1024,
                "cpu_count": os.cpu_count(),
                "affinity": sorted(os.sched_getaffinity(0)),
                "phases": phases,
                "physical_requests": events,
                "report": json.loads(Path(str(output) + ".report.json").read_text())
                if success
                else None,
                "scope": "Observations only through unchanged source-bound236 overlap entry. "
                "Original per-request60s/global1200s/generation1800s, twoJVM1GiB and "
                "candidate2threads retained. Exceptions propagate; no retry, failed-request "
                "substitution, altered scoring, timings claim or competition submission.",
            },
        )
