"""Source-bound3340 exact replay of shared original scoring and batched discovery."""

import argparse
import json
import tempfile
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml import full_gap_candidate_system as system
from casmi_ml import portable_fragment_prefetch as physical
from casmi_ml.data import write_json
from casmi_ml.generation_fragment_overlap import SharedOriginalScoring
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.research_protocol import freeze
from casmi_ml.streaming_fragment_prefetch import StreamingFragmentPrefetch


def extend_batched(*args, **kwargs):
    test = pd.read_parquet(args[0])
    full = json.loads(Path(args[2]).read_text())
    routing = pd.read_csv(args[3])
    pool = StreamingFragmentPrefetch(kwargs["fragment_deadline"])
    shared = SharedOriginalScoring(args[5], args[6])
    keys = list(test.molecule_id.unique())
    records = {row["molecule_id"]: row for row in full}

    class Recorder(PersistentMonomerMetFrag):
        def _run(self, *request):
            return pool.record(self, *request)

    class Consumer(PersistentMonomerMetFrag):
        def _run(self, *request):
            return pool.fetch(*request)

    try:
        with ExitStack() as stack:
            shared.enter(stack)
            with tempfile.TemporaryDirectory(
                prefix="casmi-overlap-control-plan-"
            ) as folder:
                folder = Path(folder)
                for start in range(0, len(keys), 16):
                    batch = keys[start : start + 16]
                    test[test.molecule_id.isin(batch)].to_parquet(
                        folder / "test.parquet", index=False
                    )
                    pd.DataFrame(
                        [
                            {
                                "molecule_id": key,
                                "smiles": ";".join(records[key]["smiles"][:25]),
                            }
                            for key in batch
                        ]
                    ).to_csv(folder / "baseline.csv", index=False)
                    write_json(folder / "full.json", [records[key] for key in batch])
                    routing[routing.molecule_id.isin(batch)].to_csv(
                        folder / "routing.csv", index=False
                    )
                    planning_args = [
                        folder / "test.parquet",
                        folder / "baseline.csv",
                        folder / "full.json",
                        folder / "routing.csv",
                        folder / "plan.csv",
                        *args[5:],
                    ]
                    planning_kwargs = dict(kwargs, output_full_rankings=None)
                    with patch(
                        "casmi_ml.metfrag_persistent.PersistentMonomerMetFrag",
                        new=Recorder,
                    ):
                        physical.serial_extend(*planning_args, **planning_kwargs)
            with patch(
                "casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=Consumer
            ):
                return physical.serial_extend(*args, **kwargs)
    finally:
        pool.close()


def run(output, model):
    output = Path(output)
    freeze(
        output / "overlap_control_sources.json",
        {
            "control_source_sha256": digest(__file__),
            "overlap_source_sha256": digest("casmi_ml/generation_fragment_overlap.py"),
            "stream_source_sha256": digest("casmi_ml/generation_handoff_stream.py"),
            "pipeline_source_sha256": digest("casmi_ml/portable_overlap_pipeline.py"),
            "rule": "Full original2000/1340 current236 ranks under shared immutable catalog/encoder/digest memo and exact original score caches plus16-completed-molecule batched physical discovery; originaltwoJVM/two candidate threads/1200physical deadline, noBLAS/model/gate/candidate changes. All3340 ranks and complete physical status required. Existing complete handoffs stand in for generation events;397 fullactual400 event identity and400 newcold400 physicaldict/rank proof separately establish event emitter and runtime. No speed/accuracy/remote acceptance.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with patch.object(system, "extend", new=extend_batched):
        report = system.run(output, model, control=True)
    if (
        not report["control_passed"]
        or report["development_budget_fallbacks"]
        or report["incremental_new_failure_contexts"]
    ):
        raise ValueError("Complete3340 exact original physical control required")
    report["overlap_control_sources_sha256"] = digest(
        output / "overlap_control_sources.json"
    )
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
