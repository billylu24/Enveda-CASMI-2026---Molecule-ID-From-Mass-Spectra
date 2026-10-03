"""Amend the same local winner with complete observable inference evidence."""

import argparse
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.low_leader_replay import SCREEN
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, development_gate


def validate_replay(report, protocol, config, historical, completed):
    if (
        not report.get("verified")
        or not report.get("local_only")
        or report.get("development_budget_fallbacks") != 0
        or report.get("incremental_new_failure_contexts") != 0
        or protocol.get("inference_sha256")
        != digest("casmi_ml/chembl_routed_inference.py")
        or protocol.get("low_leader_source_sha256")
        != digest("casmi_ml/low_leader_inference.py")
    ):
        raise ValueError("Verified current-source complete local replay required")
    selected = {mode: report[mode]["metrics"] for mode in ("unknown", "known")}
    for mode, count in [("unknown", 2000), ("known", 1340)]:
        if (
            report[mode]["molecules"] != count
            or report[mode]["full_rank_matches"] != count
        ):
            raise ValueError("Every full ranking must match on the original cohort")
    gates = {
        name: development_gate(selected, baseline, config)
        for name, baseline in [
            ("historical181", historical),
            ("completed_original", completed),
        ]
    }
    if not all(gate["eligible"] for gate in gates.values()):
        raise ValueError("Complete local inference failed original development gates")
    return selected, gates


def activate(directory):
    c = Controller()
    if not c.local_only():
        raise ValueError("Activation is local-only")
    directory = Path(directory)
    report = json.loads((directory / "report.json").read_text())
    protocol = json.loads((directory / "protocol.json").read_text())
    best = c.read()["development_best"]
    if best["round"] != SCREEN.name:
        raise ValueError("Same screened winner required for evidence amendment")
    incumbent = json.loads(
        Path(
            "artifacts/research_loop/rounds/0181_high_strong_slot_full/report.json"
        ).read_text()
    )
    control = json.loads((CONTROL / "report.json").read_text())
    selected, gates = validate_replay(
        report,
        protocol,
        c.config,
        {m: incumbent[m]["strong_slot_prefix"] for m in ("unknown", "known")},
        {m: control[m]["diagnostic_residual"] for m in ("unknown", "known")},
    )
    if protocol["control_rankings_sha256"] != digest(
        CONTROL / "selected_rankings.json"
    ) or protocol["screened_rankings_sha256"] != digest(
        SCREEN / "selected_rankings.json"
    ):
        raise ValueError("Historical evidence changed")
    # Exact match was established by the replay against independently derived
    # control-plus-low-leader ranks before any labels were evaluated. This is
    # an evidence/coverage correction of the same rule, not another optimization.
    amendment = {
        "scientific_round": SCREEN.name,
        "evidence_round": directory.name,
        "previous_metrics": best["metrics"],
        "metrics": selected,
        "gates": gates,
        "replay_report_sha256": digest(directory / "report.json"),
        "rankings_sha256": digest(directory / "selected_rankings.json"),
        "external_config_sha256": digest(directory / "external_config.json"),
        "inference_sha256": protocol["inference_sha256"],
        "low_leader_source_sha256": protocol["low_leader_source_sha256"],
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "scope": "Same screened low leader rule; use audited complete physical original-control evidence on other branches and matched candidate universe. Cached development replay, no independent acceptance or cold runtime proof.",
    }
    write_json("results/research_loop/low_leader_evidence_amendment.json", amendment)
    c.change(
        lambda state: state.update(
            development_best={
                "round": SCREEN.name,
                "variant": "supported_leader_replay",
                "directory": str(directory),
                "metrics": selected,
                "evidence_amendment": "results/research_loop/low_leader_evidence_amendment.json",
            },
            local_verified_inference={
                **amendment,
                "external_config": str(directory / "external_config.json"),
            },
        )
    )
    return amendment


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directory", required=True)
    print(json.dumps(activate(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
