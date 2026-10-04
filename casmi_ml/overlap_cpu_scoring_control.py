"""Bind unchanged complete scoring to the separately verified CPU generator worker."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.overlap_scoring_full_control import run as original_run
from casmi_ml.research_protocol import freeze

EVENTS = Path("artifacts/research_loop/rounds/0410_generation_cpu_worker400_control")
COLD = Path(
    "artifacts/research_loop/rounds/0411_overlap_generation_cpu_cold400_control"
)
SOURCES = ("generation_handoff_cpu_worker", "portable_overlap_cpu_pipeline")


def check_controls():
    events = json.loads((EVENTS / "report.json").read_text())
    cold = json.loads((COLD / "report.json").read_text())
    if not (
        events["all400_completed_events_exact"]
        and events["all_generation_csv_audit_fullhandoffs_exact"]
        and cold["valid"]
        and cold["all400_top25_exact"]
        and cold["all265_raw_physical_dictionaries_exact"]
        and cold["development_budget_fallbacks"] == 0
        and cold["seconds"] < 1800
        and cold["parent_peak_rss_mib"] < 8192
    ):
        raise ValueError(
            "Complete original400 CPU worker and physical identity required"
        )
    if events["cpu_worker_source_sha256"] != digest(
        "casmi_ml/generation_handoff_cpu_worker.py"
    ):
        raise ValueError("Verified CPU worker changed")
    for field, name in (
        ("worker_source_sha256", SOURCES[0]),
        ("pipeline_source_sha256", SOURCES[1]),
    ):
        if cold["generation_cpu_runtime"][field] != digest("casmi_ml/" + name + ".py"):
            raise ValueError("Cold CPU source changed: " + name)
    if events["original_generation_source_sha256"] != digest(
        "casmi_ml/generation_inference.py"
    ):
        raise ValueError("Original verified generator changed")
    for field, name in (
        ("stream_source_sha256", "generation_handoff_stream"),
        ("overlap_source_sha256", "generation_fragment_overlap"),
        ("pipeline_source_sha256", "portable_overlap_pipeline"),
    ):
        if cold["overlap"][field] != digest("casmi_ml/" + name + ".py"):
            raise ValueError("Original cold overlap source changed: " + name)
    return events, cold


def run(output, model):
    check_controls()
    output = Path(output)
    freeze(
        output / "cpu_control_sources.json",
        {
            "sources": {name: digest("casmi_ml/" + name + ".py") for name in SOURCES},
            "generation_control_sha256": digest(EVENTS / "report.json"),
            "cold_control_sha256": digest(COLD / "report.json"),
            "control_source_sha256": digest(__file__),
            "scope": "Unchanged original3340 scoring replay. Generator CPU4->1 is separately checked on all400 complete events, CSV and audit; no generator is run in this cached development replay. Original budgets and physical completeness retained. No speed or accuracy claim.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    report = original_run(output, model)
    report["cpu_control_sources_sha256"] = digest(output / "cpu_control_sources.json")
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", required=True)
    run(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
