"""Bind full3340 original cached scoring to separately cold-verified mass reuse."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.mass_reuse_fragment_runtime import bound_mass_reuse
from casmi_ml.metfrag import digest
from casmi_ml.overlap_cpu_scoring_control import run as original_run
from casmi_ml.research_protocol import freeze

COLD = Path("artifacts/research_loop/rounds/0424_mass_reuse_cold400_complete_handoff_control")
CLASSES = Path(
    "artifacts/research_loop/rounds/0422_mass_reuse_heavy_request_abba/derived_classes"
)


def run(output, model):
    report = json.loads((COLD / "report.json").read_text())
    if not (
        report["valid"]
        and report["all400_top25_exact"]
        and report["all400_generation_csv_audit_fullhandoffs_exact"]
        and report["all265_raw_physical_dictionaries_exact"]
        and report["development_budget_fallbacks"] == 0
    ):
        raise ValueError("Complete new cold400 mass-reuse control required")
    output = Path(output)
    expected = json.loads((COLD / "protocol.json").read_text())["classes"]
    freeze(
        output / "mass_reuse_sources.json",
        {
            "source_sha256": digest(__file__),
            "runtime_source_sha256": digest("casmi_ml/mass_reuse_fragment_runtime.py"),
            "cold_proof_sha256": digest(COLD / "report.json"),
            "classes": expected,
            "scope": "Full3340 source-bound unchanged236 cached original scoring; no generation/physical speed measurement. Separate newcold400 original265 rawdict and400generation exact evidence required; original mass/scoring semantics and deadlines preserved. No accuracy/deployment claim.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    with bound_mass_reuse(CLASSES, expected):
        result = original_run(output, model)
    result["mass_reuse_sources_sha256"] = digest(output / "mass_reuse_sources.json")
    write_json(output / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    p.add_argument("--model", required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
