"""Fresh Java verification that changed candidate pools permit exact score reuse."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import pandas as pd
from rdkit import Chem

from casmi_ml.chembl_high_strong_slots import CLASSES, JAR, JAVA
from casmi_ml.data import write_json
from casmi_ml.merged_fragments import merge_records
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.research_protocol import freeze


def reuse_scores(seed, target, missing):
    """Preserve absent scores and their semantics; never fabricate zero evidence."""
    values = {key: value for key, value in seed.items() if key in target}
    if set(values) & set(missing):
        raise ValueError("Incremental scores overlap existing completed candidates")
    if set(missing) - set(target):
        raise ValueError("Incremental result contains unrequested candidates")
    values.update(missing)
    return values


def run(output, control, proposals, target_proposals, limit=12):
    output, control = Path(output), Path(control)
    if limit != 12:
        raise ValueError("Fixed observable hash12 verification required")
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((control / "protocol.json").read_text())["config"]
    seed_proposals = json.loads(Path(proposals).read_text())
    targets = json.loads(Path(target_proposals).read_text())
    test = pd.read_parquet(control / "unknown/test.parquet")
    forbidden = {"inchikey14", "normalized_smiles", "fingerprint", "molecular_formula"}
    if forbidden & set(test):
        raise ValueError("Benchmark query inputs must be unlabeled")
    groups = test.groupby("molecule_id", sort=False).indices
    ids = sorted(
        groups,
        key=lambda k: hashlib.sha256(
            ("incremental-physical-20261003:" + k).encode()
        ).digest(),
    )[:limit]
    original = {
        r["molecule_id"]: r["smiles"]
        for r in json.loads((control / "unknown/baseline_full.json").read_text())
    }
    catalog = pd.read_parquet(
        config["catalog"], columns=["inchikey14", "normalized_smiles"]
    )
    needed = {k for key in ids for k in seed_proposals[key] + targets[key]}
    lookup = (
        catalog[catalog.inchikey14.isin(needed)]
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    del catalog
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "control_protocol_sha256": digest(control / "protocol.json"),
            "test_sha256": digest(control / "unknown/test.parquet"),
            "baseline_full_sha256": digest(control / "unknown/baseline_full.json"),
            "seed_proposals_sha256": digest(proposals),
            "target_proposals_sha256": digest(target_proposals),
            "catalog_sha256": digest(config["catalog"]),
            "engine": {
                str(p): digest(p)
                for p in [
                    JAVA,
                    JAR,
                    CLASSES / "MetFragWorker.class",
                    CLASSES
                    / "de/ipbhalle/metfraglib/process/CombinedMetFragProcess.class",
                ]
            },
            "limit": limit,
            "selector": "First hash256 incremental-physical-20261003 query IDs",
            "fragment_seconds": 1200,
            "request_seconds": 60,
            "candidate_threads": 2,
            "polling_ms": 10,
            "rule": "Same observable merged records and100 candidate cutoff. Independent fresh full seed/precise-target pools, versus fresh scores only for new keys or changed SMILES plus exact previous candidate scores. Each per-spectrum dictionary and supported status must match exactly. No query truth or metric selection.",
            "diagnostic_only": True,
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    engines = {
        name: PersistentMonomerMetFrag(
            JAR,
            output / (name + "_fresh_cache"),
            java=str(JAVA),
            classes=CLASSES,
            threads=2,
        )
        for name in ("seed", "full", "incremental")
    }
    if any(list(engine.cache.glob("*.json")) for engine in engines.values()):
        raise ValueError("Benchmark requires independent empty physical caches")
    started = time.monotonic()
    deadline = started + 1200
    counts = {
        "queries": 0,
        "spectra": 0,
        "exact_dictionaries": 0,
        "complete": 0,
        "unsupported_or_empty": 0,
        "failures": 0,
        "new_candidates": 0,
        "reused_candidates": 0,
    }
    timings = {name: 0.0 for name in engines}
    private = []
    try:
        for key in ids:
            prior = {
                Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14]: s for s in original[key]
            }
            first = next(iter(prior))
            old = {
                k: lookup[k]
                for k in [k for k in seed_proposals[key] if k not in prior][:100]
            }
            target = {
                k: lookup[k] for k in [k for k in targets[key] if k not in prior][:100]
            }
            old[first] = target[first] = prior[first]
            missing = {k: s for k, s in target.items() if old.get(k) != s}
            reusable = {k for k, s in target.items() if old.get(k) == s}
            counts["new_candidates"] += len(missing)
            counts["reused_candidates"] += len(reusable)
            for row in merge_records(
                test.iloc[groups[key]].drop(columns="molecule_id").to_dict("records")
            ):
                results = {}
                for name, candidates in (
                    ("seed", old),
                    ("full", target),
                    ("incremental", missing),
                ):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Shared benchmark fragment budget exhausted")
                    if not candidates:
                        results[name] = {"status": "complete", "scores": {}}
                        continue
                    engines[name].timeout = min(60, deadline - time.monotonic())
                    before = time.monotonic()
                    results[name] = engines[name].score(row, candidates)
                    timings[name] += time.monotonic() - before
                seed = {
                    k: v for k, v in results["seed"]["scores"].items() if k in reusable
                }
                reconstructed = reuse_scores(
                    seed, target, results["incremental"]["scores"]
                )
                equal = reconstructed == results["full"]["scores"]
                compatible = results["seed"]["status"] == results["full"][
                    "status"
                ] and (
                    not missing
                    or results["incremental"]["status"] == results["full"]["status"]
                )
                failed = any(
                    v["status"]
                    not in ("complete", "unsupported_ion_or_resolution", "empty")
                    for v in results.values()
                )
                counts["spectra"] += 1
                counts["exact_dictionaries"] += equal
                counts["complete"] += results["full"]["status"] == "complete"
                counts["unsupported_or_empty"] += results["full"]["status"] in (
                    "unsupported_ion_or_resolution",
                    "empty",
                )
                counts["failures"] += failed or not compatible or not equal
                private.append(
                    {
                        "key": key,
                        "equal": equal,
                        "compatible_status": compatible,
                        "statuses": {
                            name: value["status"] for name, value in results.items()
                        },
                    }
                )
            counts["queries"] += 1
            print(
                "incremental benchmark",
                counts["queries"],
                counts["spectra"],
                "failures",
                counts["failures"],
                flush=True,
            )
    finally:
        for engine in engines.values():
            engine.close()
    report = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "verified": counts["failures"] == 0
        and counts["queries"] == limit
        and counts["complete"] > 0,
        "counts": counts,
        "seconds_by_arm": timings,
        "seconds_total": time.monotonic() - started,
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "scope": "Fresh physics scoring benchmark only; no cold full-system or accuracy claim. Seed preparation excluded from incremental runtime and reported separately.",
    }
    write_json(output / "private_rows.json", private)
    write_json(output / "report.json", report)
    if not report["verified"]:
        raise ValueError("Candidate-pool-independent physical reuse not verified")
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--control", required=True)
    p.add_argument("--proposals", required=True)
    p.add_argument("--target-proposals", required=True)
    p.add_argument("--limit", type=int, default=12)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
