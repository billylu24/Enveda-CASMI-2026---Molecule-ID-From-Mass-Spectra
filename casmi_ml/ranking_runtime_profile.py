"""Fixed opaque-query runtime profile with exact local full-ranking control."""

import argparse
import cProfile
import hashlib
import io
import json
import pstats
import time
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml.chembl_routed_inference import extend
from casmi_ml.data import write_json
from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.diagnostic_control_audit import structure_key
from casmi_ml.diagnostic_system_evaluation import resolve_local_incremental
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze

VERIFIED = Path("artifacts/research_loop/rounds/0237_low_leader_full_unlabeled_replay")


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Runtime profile is local-only")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    config = json.loads((VERIFIED / "external_config.json").read_text())
    expected = json.loads((VERIFIED / "selected_rankings.json").read_text())
    benchmark, inherited, _ = resolve_local_incremental(str(CONTROL), None)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "inference_source_sha256": digest("casmi_ml/chembl_routed_inference.py"),
            "verified_rankings_sha256": digest(VERIFIED / "selected_rankings.json"),
            "sampling": "100 nonempty baseline opaque identifiers per mode, SHA256 runtime-profile:+identifier order; no truth-based selection",
            "local_only": True,
            "independent_acceptance": False,
            "scope": "cProfile cached exact-rank development sample; profiling overhead is present, no cold runtime or accuracy claim",
        },
    )
    reports = {}
    engines = []

    def engine(*args, **kwargs):
        obj = DevelopmentIncrementalMetFrag(*args, benchmark=benchmark, **kwargs)
        engines.append(obj)
        return obj

    for mode in ("unknown", "known"):
        source = CONTROL / mode
        dest = output / mode
        dest.mkdir(exist_ok=True)
        full = json.loads((source / "baseline_full.json").read_text())
        identifiers = sorted(
            [r["molecule_id"] for r in full],
            key=lambda key: hashlib.sha256(
                ("runtime-profile:" + key).encode()
            ).digest(),
        )[:100]
        selected = set(identifiers)
        test = pd.read_parquet(source / "test.parquet")
        test[test.molecule_id.isin(selected)].to_parquet(
            dest / "test.parquet", index=False
        )
        write_json(
            dest / "full.json", [r for r in full if r["molecule_id"] in selected]
        )
        for name in ("baseline.csv", "routing.csv"):
            frame = pd.read_csv(source / name)
            frame[frame.molecule_id.isin(selected)].to_csv(dest / name, index=False)
        profiler = cProfile.Profile()
        started = time.monotonic()
        profiler.enable()
        with patch("casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=engine):
            extend(
                dest / "test.parquet",
                dest / "baseline.csv",
                dest / "full.json",
                dest / "routing.csv",
                dest / "selected.csv",
                Path("."),
                config,
                deadline=started + 3600,
                fragment_deadline=started + 1200,
                java="external/metfrag/java21/jdk-21.0.12.1+1-jre/bin/java",
                cache="artifacts/chemistry_20261001/chembl_metfrag_cache",
                output_full_rankings=dest / "selected_full.json",
                reference=ReferenceIndex(
                    "artifacts/research_loop/rounds/0001_mass_v2/reference/" + mode
                ),
            )
        profiler.disable()
        actual = {
            r["molecule_id"]: [structure_key(s) for s in r["smiles"]]
            for r in json.loads((dest / "selected_full.json").read_text())
        }
        mismatches = [key for key in identifiers if actual[key] != expected[mode][key]]
        if mismatches:
            raise ValueError("Profiling sample differs from verified full ranking")
        for sort in ("cumulative", "tottime"):
            stream = io.StringIO()
            stats = pstats.Stats(profiler, stream=stream).sort_stats(sort)
            stats.print_stats(35)
            (dest / ("profile_" + sort + ".txt")).write_text(stream.getvalue())
        stats = pstats.Stats(profiler)
        top = sorted(stats.stats.items(), key=lambda item: item[1][3], reverse=True)[
            :20
        ]
        functions = [
            {
                "file": Path(key[0]).name,
                "line": key[1],
                "function": key[2],
                "calls": value[1],
                "self_seconds": value[2],
                "cumulative_seconds": value[3],
            }
            for key, value in top
        ]
        reports[mode] = {
            "molecules": len(identifiers),
            "full_rank_matches": len(identifiers),
            "seconds_with_profile": time.monotonic() - started,
            "top_cumulative_functions": functions,
        }
        print(mode, reports[mode], flush=True)
    failures = set().union(*(e.failed_physical_requests for e in engines))
    reports.update(
        diagnostic_only=True,
        local_only=True,
        independent_acceptance=False,
        new_failure_contexts=len(failures - inherited),
    )
    if failures - inherited:
        raise ValueError("New profiling physical failure")
    write_json(output / "report.json", reports)
    return reports


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
