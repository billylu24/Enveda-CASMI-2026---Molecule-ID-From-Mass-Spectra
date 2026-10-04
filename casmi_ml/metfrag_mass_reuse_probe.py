"""Private paired heavy-request probe of peak-loop-local exact mass reuse."""

import argparse
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_jfr_observation import JAR, ORIGINAL
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

BASE_CLASSES = Path(
    "kaggle_release_verified_best_0236_overlap_cpu_once/bundle/java_classes"
)
DERIVED = Path("artifacts/research_loop/mass_reuse_private_classes")
PATCHER = Path("casmi_ml/java/MetFragMassReusePatcher.java")


def run(output):
    if not Controller().local_only():
        raise ValueError("Local-only mode required")
    output = Path(output).absolute()
    output.mkdir(parents=True, exist_ok=False)
    classes = output / "derived_classes"
    shutil.copytree(BASE_CLASSES, classes)
    for path in DERIVED.rglob("*.class"):
        target = classes / path.relative_to(DERIVED)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    events = json.loads((ORIGINAL / "request_events.json").read_text())
    selected = sorted(
        (
            e
            for e in events
            if e["status"] == "complete"
            and (
                ORIGINAL / "private_request_inputs" / f"{e['request_digest']}.json"
            ).exists()
        ),
        key=lambda e: (-e["seconds"], e["request_digest"]),
    )[:10]
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "patcher_sha256": digest(PATCHER),
            "jar_sha256": digest(JAR),
            "original_events_sha256": digest(ORIGINAL / "request_events.json"),
            "selected_inputs_sha256": [
                digest(
                    ORIGINAL / "private_request_inputs" / f"{e['request_digest']}.json"
                )
                for e in selected
            ],
            "derived_classes": {
                str(p.relative_to(classes)): digest(p) for p in classes.rglob("*.class")
            },
            "rule": "Fixed baseline/reuse/reuse/baseline ABBA, fresh ten slowest captured original407 requests each;4CPU/twoJVM/two candidate threads/Java21/defaultGC/1GiB/60s. Original pinned fragmentation/scoring methods retained instruction-identical; only added match overload receives same ordered atom mass sum once per unchanged fragment peak-loop, recalculated on each loop entry. No persistent mass memoization, peak/candidate trimming, floating tolerance or retry. All40 raw dictionaries must be exact before interpreting local paired timing. Not full400 or platform speed proof; no activation/submission.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    java = str(next(Path("external/metfrag/java21").glob("*/bin/java")).absolute())
    arms = []
    for trial, arm in enumerate(["baseline", "reuse", "reuse", "baseline"]):

        def worker(index, trial=trial, arm=arm):
            work = output / f"{trial}_{arm}" / f"worker{index}"
            engine = PersistentMonomerMetFrag(
                JAR,
                work / "fresh_cache",
                classes=BASE_CLASSES if arm == "baseline" else classes,
                java=java,
                threads=2,
                timeout=60,
            )
            rows = []
            try:
                for event in selected[index::2]:
                    payload = json.loads(
                        (
                            ORIGINAL
                            / "private_request_inputs"
                            / f"{event['request_digest']}.json"
                        ).read_text()
                    )
                    start = time.monotonic()
                    result = engine._run(
                        work / "fresh_cache" / f"{payload['cache_key']}.json",
                        payload["candidates"],
                        payload["peaks"],
                        payload["mass"],
                        payload["adduct"],
                    )
                    elapsed = time.monotonic() - start
                    expected = json.loads(
                        (
                            ORIGINAL
                            / "submission.csv.external_fragment_cache"
                            / f"{payload['cache_key']}.threads2.json"
                        ).read_text()
                    )
                    rows.append(
                        {
                            "request_digest": event["request_digest"],
                            "seconds": elapsed,
                            "status": result["status"],
                            "raw_dictionary_exact": result == expected,
                        }
                    )
                    write_json(work / "private_observations.json", rows)
                    if result != expected:
                        raise ValueError(f"{arm} original raw dictionary differs")
            finally:
                engine.close()
            return rows

        start = time.monotonic()
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = [pool.submit(worker, i) for i in range(2)]
            rows = [r for f in futures for r in f.result()]
        arms.append(
            {
                "arm": arm,
                "wall_seconds": time.monotonic() - start,
                "requests": len(rows),
                "raw_dictionary_matches": sum(r["raw_dictionary_exact"] for r in rows),
            }
        )
        print(json.dumps(arms[-1]), flush=True)
    means = {
        a: sum(r["wall_seconds"] for r in arms if r["arm"] == a) / 2
        for a in ["baseline", "reuse"]
    }
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "arms": arms,
        "mean_wall_seconds": means,
        "reuse_over_baseline_ratio": means["reuse"] / means["baseline"],
        "scope": "Matched ten-request4CPU local ABBA observation only; all40 raw dictionaries exact. No full400/source/resources/platform acceptance or accuracy improvement. Private derived runtime not activated.",
    }
    write_json(output / "report.json", report)
    print(json.dumps(report), flush=True)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
