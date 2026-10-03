"""Local-only, benchmark-bound reuse of raw per-candidate physical fragment scores."""

import fcntl
import hashlib
import json
import math
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag


def context_identity(jar_sha, peaks, mass, adduct, depth, runtime):
    payload = {
        "jar": jar_sha,
        "peaks": peaks,
        "mass": mass,
        "adduct": adduct,
        "ppm": 10,
        "absolute_da": 0.002,
        "depth": depth,
        "runtime": runtime,
        "score": "raw_FragmenterScore",
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def candidate_identity(key, smiles):
    return hashlib.sha256(json.dumps([key, smiles]).encode()).hexdigest()


def verify_complete(result, candidates):
    if result.get("status") != "complete":
        raise ValueError("Only complete physical results can enter reusable evidence")
    if set(result["scores"]) - set(candidates) or any(
        not math.isfinite(v) for v in result["scores"].values()
    ):
        raise ValueError(
            "Completed score dictionary contains invalid candidate evidence"
        )


class DevelopmentIncrementalMetFrag(PersistentMonomerMetFrag):
    def __init__(self, *args, benchmark, **kwargs):
        benchmark = Path(benchmark)
        report = json.loads((benchmark / "report.json").read_text())
        protocol = json.loads((benchmark / "protocol.json").read_text())
        if (
            not report.get("verified")
            or report["counts"]["failures"]
            or report["counts"]["exact_dictionaries"] != report["counts"]["spectra"]
        ):
            raise ValueError("Exact fresh incremental benchmark required")
        super().__init__(*args, **kwargs)
        classes = Path(kwargs["classes"])
        runtime = {
            str(classes / name): digest(classes / name)
            for name in (
                "MetFragWorker.class",
                "de/ipbhalle/metfraglib/process/CombinedMetFragProcess.class",
            )
        }
        if self.threads != 2 or protocol["engine"].get(str(self.jar)) not in (
            None,
            self.sha256,
        ):
            raise ValueError("Incremental development execution differs from benchmark")
        # Benchmark paths are relative; compare pinned runtime assets by file name.
        for file, sha in [
            (self.jar, self.sha256),
            *[(Path(p), sha) for p, sha in runtime.items()],
        ]:
            if not any(
                (
                    Path(p).name == file.name
                    or (file == self.jar and Path(p).suffix == ".jar")
                )
                and value == sha
                for p, value in protocol["engine"].items()
            ):
                raise ValueError(
                    "Incremental runtime asset differs from verified benchmark"
                )
        java_values = [
            value for p, value in protocol["engine"].items() if Path(p).name == "java"
        ]
        java_sha = digest(self.java) if java_values else None
        if java_values and java_sha not in java_values:
            raise ValueError("Incremental Java runtime differs from verified benchmark")
        self.runtime = {
            "threads": self.threads,
            "worker_and_process": runtime,
            "java_sha256": java_sha,
        }

        self.benchmark_sha256 = digest(benchmark / "report.json")
        self.index = self.cache / "development_incremental_evidence_v1"
        self.index.mkdir(exist_ok=True)
        self.require_existing = False
        self.incremental_stats = {
            "seed_full_requests": 0,
            "reused_candidates": 0,
            "new_candidates": 0,
            "incremental_requests": 0,
            "failed_requests": 0,
        }

    def _run(self, cache_path, candidates, peaks, mass, adduct):
        complete_path = cache_path.with_name(cache_path.stem + ".threads2.json")
        if self.require_existing and not complete_path.exists():
            return {"status": "not_seeded", "scores": {}}
        context = context_identity(
            self.sha256, peaks, mass, adduct, self.depth, self.runtime
        )
        index_path = self.index / (context + ".json")
        with (self.index / (context + ".lock")).open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            evidence = (
                json.loads(index_path.read_text())
                if index_path.exists()
                else {
                    "context": context,
                    "benchmark_sha256": self.benchmark_sha256,
                    "candidates": {},
                }
            )
            if (
                evidence["context"] != context
                or evidence["benchmark_sha256"] != self.benchmark_sha256
            ):
                raise ValueError("Incremental evidence context or verification differs")
            if complete_path.exists():
                result = json.loads(complete_path.read_text())
                verify_complete(result, candidates)
                if result.get("jar_sha256") != self.sha256:
                    raise ValueError("Completed physical cache engine digest differs")
                missing = candidates
                self.incremental_stats["seed_full_requests"] += 1
                provenance = digest(complete_path)
            else:
                missing = {
                    k: s
                    for k, s in candidates.items()
                    if candidate_identity(k, s) not in evidence["candidates"]
                }
                reused = len(candidates) - len(missing)
                self.incremental_stats["reused_candidates"] += reused
                self.incremental_stats["new_candidates"] += len(missing)
                if missing:
                    payload = {
                        "jar": self.sha256,
                        "peaks": peaks,
                        "mass": mass,
                        "adduct": adduct,
                        "candidates": sorted(missing.items()),
                        "ppm": 10,
                        "absolute_da": 0.002,
                        "depth": self.depth,
                    }
                    partial = self.cache / (
                        hashlib.sha256(
                            json.dumps(payload, sort_keys=True).encode()
                        ).hexdigest()
                        + ".json"
                    )
                    result = super()._run(partial, missing, peaks, mass, adduct)
                    self.incremental_stats["incremental_requests"] += 1
                    if result.get("status") != "complete":
                        self.incremental_stats["failed_requests"] += 1
                        return result
                    verify_complete(result, missing)
                    provenance = digest(
                        partial.with_name(partial.stem + ".threads2.json")
                    )
                else:
                    result = {"status": "complete", "scores": {}}
                    provenance = None
            for key, smiles in missing.items():
                candidate = candidate_identity(key, smiles)
                entry = {
                    "score": result["scores"].get(key),
                    "physical_result_sha256": provenance,
                }
                previous = evidence["candidates"].get(candidate)
                if previous is not None and previous["score"] != entry["score"]:
                    raise ValueError(
                        "Same candidate physical score differs across completed pools"
                    )
                evidence["candidates"][candidate] = entry
            write_json(index_path, evidence)
            scores = {
                key: evidence["candidates"][candidate_identity(key, smiles)]["score"]
                for key, smiles in candidates.items()
                if evidence["candidates"][candidate_identity(key, smiles)]["score"]
                is not None
            }
            full_result = {
                "status": "complete",
                "scores": scores,
                "jar_sha256": self.sha256,
            }
            if not complete_path.exists():
                full_result["development_incremental_reuse"] = {
                    "benchmark_sha256": self.benchmark_sha256,
                    "context": context,
                    "evidence_sha256": digest(index_path),
                }
                write_json(complete_path, full_result)
            return full_result
