"""Observe unchanged heavy MetFrag requests with external JVM sampling only."""

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

ORIGINAL = Path(
    "artifacts/research_loop/rounds/0407_overlap_request_local400_observation"
)
JAR = Path("external/metfrag/MetFragCommandLine-2.6.11.jar")
CLASSES = Path("artifacts/research_loop/metfrag_worker_classes")


def aggregate_samples(events):
    """Export method-level counts only, never request contents or spectrum IDs."""
    frames = Counter()
    stacks = Counter()
    states = Counter()
    allocations = Counter()
    gc_seconds = 0.0
    execution_samples = 0
    for event in events:
        kind = event["type"]
        values = event["values"]
        if kind == "jdk.ExecutionSample":
            execution_samples += 1
            states[values.get("state", "unknown")] += 1
            trace = values.get("stackTrace") or {}
            methods = []
            for frame in trace.get("frames", []):
                method = frame["method"]
                owner = method["type"]["name"].replace("/", ".")
                methods.append(owner + "." + method["name"])
            if methods:
                frames[methods[0]] += 1
                # Inclusive stack membership, not multiple counts for recursion.
                stacks.update(set(methods))
        elif kind == "jdk.ObjectAllocationSample":
            owner = values["objectClass"]["name"].replace("/", ".")
            allocations[owner] += values["weight"]
        elif kind == "jdk.GarbageCollection":
            # JFR JSON durations are ISO8601 seconds for these short events.
            duration = values["duration"]
            if not duration.startswith("PT") or not duration.endswith("S"):
                raise ValueError("Unexpected JFR duration encoding")
            gc_seconds += float(duration[2:-1])
    return {
        "execution_samples": execution_samples,
        "sampled_thread_states": dict(states),
        "top_leaf_methods": frames.most_common(30),
        "top_inclusive_methods": stacks.most_common(30),
        "top_allocation_sample_weights_bytes": allocations.most_common(20),
        "gc_event_duration_sum_seconds": gc_seconds,
        "scope": "Sample counts and sampled allocation weights, not exact CPU time or allocated bytes. GC duration sums may overlap work. Profiling is diagnostic only, not paired speed evidence.",
    }


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Persistent local-only mode required")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    java = next(Path("external/metfrag/java21").glob("*/bin/java")).absolute()
    jcmd, jfr = shutil.which("jcmd"), shutil.which("jfr")
    if not jcmd or not jfr:
        raise ValueError("JFR observation tools unavailable")
    events = json.loads((ORIGINAL / "request_events.json").read_text())
    selected = sorted(
        (
            e
            for e in events
            if (
                ORIGINAL / "private_request_inputs" / f"{e['request_digest']}.json"
            ).exists()
            and e["status"] == "complete"
        ),
        key=lambda e: (-e["seconds"], e["request_digest"]),
    )[:10]
    if len(selected) != 10:
        raise ValueError("Prespecified ten original heavy requests unavailable")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "original_events_sha256": digest(ORIGINAL / "request_events.json"),
            "selected_private_inputs_sha256": [
                digest(
                    ORIGINAL / "private_request_inputs" / f"{e['request_digest']}.json"
                )
                for e in selected
            ],
            "jar_sha256": digest(JAR),
            "worker_sha256": digest(CLASSES / "MetFragWorker.class"),
            "java_sha256": digest(java),
            "rule": "Original407 ten slowest complete requests with captured private planning inputs; deterministic descending duration/hash tiebreak, alternate between two JVMs. Original Java21/defaultGC/1GiB heap/2candidate threads/depth2/60second request timeout. 4CPU affinity set by driver. Fresh caches; every complete raw dictionary must equal original exactly, no tolerance/retry/omission. External JFR profile only. Sampled timings are not speed proof. No candidate/peak/model changes, no remote publication or query input packaging.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )

    def worker(index):
        work = output / f"worker{index}"
        engine = PersistentMonomerMetFrag(
            JAR,
            work / "fresh_cache",
            classes=CLASSES,
            java=str(java),
            threads=2,
            timeout=60,
        )
        engine.runner.command.insert(
            1, "-XX:StartFlightRecording=name=casmi,settings=profile,disk=true"
        )
        records = []
        try:
            for event in selected[index::2]:
                path = (
                    ORIGINAL
                    / "private_request_inputs"
                    / f"{event['request_digest']}.json"
                )
                payload = json.loads(path.read_text())
                started = time.monotonic()
                result = engine._run(
                    work / "fresh_cache" / f"{payload['cache_key']}.json",
                    payload["candidates"],
                    payload["peaks"],
                    payload["mass"],
                    payload["adduct"],
                )
                elapsed = time.monotonic() - started
                expected = json.loads(
                    (
                        ORIGINAL
                        / "submission.csv.external_fragment_cache"
                        / f"{payload['cache_key']}.threads2.json"
                    ).read_text()
                )
                record = {
                    "request_digest": event["request_digest"],
                    "seconds": elapsed,
                    "status": result["status"],
                    "raw_dictionary_exact": result == expected,
                    "candidates": event["candidates"],
                    "peaks": event["peaks"],
                }
                records.append(record)
                write_json(work / "private_request_observations.json", records)
                print(
                    f"JFR worker{index}: complete={len(records)}, exact={record['raw_dictionary_exact']}, seconds={elapsed:.3f}",
                    flush=True,
                )
                if result["status"] != "complete" or result != expected:
                    raise ValueError(
                        "Profiled original physical dictionary is not exact"
                    )
            # jcmd parses its own argument string; dump into a space-free path.
            with tempfile.TemporaryDirectory(prefix="casmi-jfr-dump-") as temp:
                target = Path(temp) / "original_requests.jfr"
                dump = subprocess.run(
                    [
                        jcmd,
                        str(engine.runner.process.pid),
                        "JFR.dump",
                        "name=casmi",
                        f"filename={target}",
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                    timeout=30,
                )
                (work / "jcmd_dump.log").write_text(dump.stdout + dump.stderr)
                if not target.is_file():
                    raise ValueError("JFR.dump did not produce the requested recording")
                shutil.copy2(target, work / "original_requests.jfr")
        finally:
            engine.close()
        with (work / "private_jfr_events.json").open("w") as stream:
            subprocess.run(
                [
                    jfr,
                    "print",
                    "--json",
                    "--events",
                    "jdk.ExecutionSample,jdk.ObjectAllocationSample,jdk.GarbageCollection",
                    str(work / "original_requests.jfr"),
                ],
                check=True,
                stdout=stream,
                timeout=60,
            )
        return records, json.loads((work / "private_jfr_events.json").read_text())[
            "recording"
        ]["events"]

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(worker, i) for i in range(2)]
        results = [f.result() for f in futures]
    records = [r for rows, _ in results for r in rows]
    samples = [e for _, events in results for e in events]
    write_json(output / "private_request_observations.json", records)
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "complete_requests": len(records),
        "raw_dictionaries_exact": sum(r["raw_dictionary_exact"] for r in records),
        "original_candidate_threads": 2,
        "cpu_affinity_count": len(__import__("os").sched_getaffinity(0)),
        "profiled_wall_seconds": time.monotonic() - started,
        "jfr": aggregate_samples(samples),
        "scope": "Original heavy-request hotspot observation only; no algorithm, deadlines or model change, no full400 verification or optimization claim. All request contents and JFR events stay private locally.",
    }
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
