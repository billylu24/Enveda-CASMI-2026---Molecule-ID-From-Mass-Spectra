"""Fresh-cache real Java score equivalence and timing for sequential JVM reuse."""

import argparse
import hashlib
import json
import resource
import time
from pathlib import Path

import pandas as pd

from casmi_ml.chembl_catalog import DERIVED
from casmi_ml.chembl_critic_slots import score_cache_key
from casmi_ml.chembl_high_fragment_pilot import HIGH, fragment_structures
from casmi_ml.chemistry import clean_peaks, high_resolution
from casmi_ml.chemistry_experiment import candidate_lookup
from casmi_ml.data import write_json
from casmi_ml.generated_score_combination import GENERATED
from casmi_ml.merged_fragments import score_group_merged
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_monomer import ALIASES, MonomerMetFrag
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.research_protocol import ROOT, freeze

INCUMBENT = Path("artifacts/research_loop/rounds/0149_chembl_routed_combination")
JAR = Path("external/metfrag/MetFragCommandLine-2.6.11.jar")
JAVA = Path("external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java")
CLASSES = Path("artifacts/research_loop/metfrag_worker_classes")


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "persistent_source_sha256": digest("casmi_ml/metfrag_persistent.py"),
            "worker_source_sha256": digest("casmi_ml/java/MetFragWorker.java"),
            "worker_bytecode_sha256": digest(CLASSES / "MetFragWorker.class"),
            "jar_sha256": digest(JAR),
            "java_sha256": digest(JAVA),
            "high_protocol_sha256": digest(HIGH / "protocol.json"),
            "high_scores_sha256": digest(HIGH / "critic_scores.json"),
            "incumbent_rankings_sha256": digest(INCUMBENT / "selected_rankings.json"),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "limit": 12,
            "rule": "First12 hash200 already inserted unknown high groups with observable supported precise nonempty monomer input; same original candidate100+actual first/merged spectra/depth2; old CLI vs persistent same pinned jar; separate fresh caches",
            "stage_seconds": 600,
            "jvm_heap_mib": 1024,
            "request_timeout": 60,
            "new_training": False,
            "new_sampling": False,
            "truth_used": False,
            "cohort": "repeated_development",
            "independent_acceptance": False,
        },
    )
    if any((output / name).exists() for name in ("cli_cache", "persistent_cache")):
        raise ValueError("Benchmark requires a fresh output root, no cache reuse")
    frame = pd.read_parquet(ROOT / "researchdev.parquet")
    groups = frame.groupby("inchikey14", sort=True).indices
    selected = json.loads((INCUMBENT / "selected_rankings.json").read_text())["unknown"]
    original = json.loads((INCUMBENT / "low_rankings.json").read_text())["baseline"][
        "unknown"
    ]
    conf = {
        r["key"]: r["confidence"]
        for r in json.loads((ROOT / "researchdev_unknown_chemical.json").read_text())
    }
    config = json.loads((HIGH / "protocol.json").read_text())
    proposals_path = Path(config["proposal_path"])
    if digest(proposals_path) != config["native_proposals_sha256"]:
        raise ValueError("High proposal source changed")
    proposals = json.loads(proposals_path.read_text())
    pairs = json.loads((HIGH / "critic_scores.json").read_text())
    binding = {k: config[k] for k in ("encoder_sha256", "critic_sha256")}
    external = (
        pd.read_parquet(DERIVED, columns=["inchikey14", "normalized_smiles"])
        .set_index("inchikey14")
        .normalized_smiles.to_dict()
    )
    lookup = candidate_lookup(ROOT, "researchdev", "unknown")
    pubchem = pd.read_parquet(
        "external/pubchemlite/structures.parquet",
        columns=["inchikey14", "normalized_smiles"],
    )
    lookup.update(
        {
            r.inchikey14: r.normalized_smiles
            for r in pubchem.itertuples()
            if r.inchikey14 not in lookup
        }
    )
    generated = {
        r["key"]: {c["key"]: c["smiles"] for c in r["candidates"]}
        for r in json.loads(GENERATED.read_text())
    }
    cases = []
    keys = sorted(
        groups,
        key=lambda k: hashlib.sha256(
            ("fragment-representative-20261003:" + k).encode()
        ).digest(),
    )[:200]
    for key in keys:
        if conf[key] < 0.5 or selected[key] == original[key]:
            continue
        records = (
            frame.iloc[groups[key]]
            .drop(
                columns=[
                    c
                    for c in (
                        "inchikey14",
                        "normalized_smiles",
                        "molecular_formula",
                        "fingerprint",
                    )
                    if c in frame
                ]
            )
            .to_dict("records")
        )
        if not any(
            r.get("adduct") in ALIASES
            and high_resolution(r.get("instrument_type"))
            and len(clean_peaks(r)[0])
            for r in records
        ):
            continue
        prior = original[key]
        native = [c for c in proposals[key] if c not in set(prior)][:100]
        scores = pairs[score_cache_key(key, prior[0], native, binding)]
        candidates = sorted(native, key=lambda c: (-scores[c], c))
        structures = fragment_structures(
            key, prior[0], candidates, external, lookup, generated
        )
        cases.append((key, records, structures))
        if len(cases) == 12:
            break
    if len(cases) != 12:
        raise ValueError("Insufficient observable supported benchmark cases")
    old = MonomerMetFrag(JAR, output / "cli_cache", java=str(JAVA))
    new = PersistentMonomerMetFrag(
        JAR, output / "persistent_cache", java=str(JAVA), classes=CLASSES
    )
    results, times, statuses = {}, {}, {}
    deadline = time.monotonic() + 600
    try:
        for name, engine in (("cli", old), ("persistent", new)):
            started = time.monotonic()
            results[name] = {}
            for i, (key, records, structures) in enumerate(cases, 1):
                scores, fallback = score_group_merged(
                    engine, records, structures, deadline=deadline
                )
                if fallback:
                    raise TimeoutError("Benchmark stage budget exhausted")
                results[name][key] = scores
                print(name, i, "seconds", time.monotonic() - started, flush=True)
            times[name] = time.monotonic() - started
            statuses[name] = engine.status_counts
    finally:
        new.close()
    if any(
        status.get("timeout", 0) or status.get("failed", 0)
        for status in statuses.values()
    ):
        raise ValueError("Failed request invalidates equivalence/timing benchmark")
    if results["cli"] != results["persistent"]:
        write_json(output / "private_score_difference.json", results)
        raise ValueError("Persistent JVM scores differ from command line")
    report = {
        "diagnostic_only": True,
        "molecules": len(cases),
        "exact_score_dictionary_matches": len(cases),
        "fresh_cache_per_arm": True,
        "seconds": times,
        "speedup": times["cli"] / times["persistent"],
        "spectrum_status_counts": statuses,
        "parent_peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        / 1024,
        "independent_acceptance": False,
        "scope": "Runtime benchmark only; no new accuracy/release claim",
    }
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output), indent=2))


if __name__ == "__main__":
    main()
