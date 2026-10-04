"""Run exact original-slot control before the single frozen local five-slot arm."""

import fcntl
import json
from pathlib import Path

from casmi_ml.candidate_rank_movement_audit import run as movement
from casmi_ml.data import write_json
from casmi_ml.high_slot_depth_system import VARIANT
from casmi_ml.local_research_snapshot import refresh
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, development_gate
from casmi_ml.research_protocol import freeze

MODEL = Path(
    "artifacts/research_loop/rounds/0354_fragment_prefetch_zero_control/zero_model.json"
)


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Slot-depth followup is local-only")
    with (c.root / "high_slot_depth_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        best = c.read()["development_best"]
        freeze(
            c.root / "high_slot_depth_preregistration.json",
            {
                "source_sha256": digest(__file__),
                "selector_source_sha256": digest("casmi_ml/high_slot_depth_system.py"),
                "audit_sha256": digest(
                    "results/research_loop/candidate_stage_retention.json"
                ),
                "model_sha256": digest(MODEL),
                "incumbent": best,
                "hypothesis": "Exact stage audit finds72 available unknown truths present in actual100 proposal shortlist but absent finally;23 occur in high_strong_slot_inserted groups. Previous183 scans100 yet caps3 slots; this new arm caps5 within first5 with the same individual gates and same leader/prefix. Aggregate diagnosis motivates one frozen experiment, no truth-based candidate selection.",
                "sequence": [
                    "0355_high_slot_depth_three_control",
                    "0356_high_slot_depth_five_system",
                ],
                "local_only": True,
                "remote_publication": False,
                "independent_acceptance": False,
                "holdout_used": False,
            },
        )
        reports = {}
        for slots, rid in [
            (3, "0355_high_slot_depth_three_control"),
            (5, "0356_high_slot_depth_five_system"),
        ]:
            if c.stopped():
                return
            out = c.root / "rounds" / rid
            c.register_round(
                rid,
                "implementation_control" if slots == 3 else "candidate_slot_depth",
                [
                    ".venv/bin/python",
                    "-u",
                    "-m",
                    "casmi_ml.high_slot_depth_system",
                    "--output",
                    str(out),
                    "--model",
                    str(MODEL),
                    "--slots",
                    str(slots),
                ],
                out / "report.json",
            )
            if not (out / "report.json").exists():
                c.run_auxiliary(rid)
            report = json.loads((out / "report.json").read_text())
            if (
                report["development_budget_fallbacks"]
                or report["incremental_new_failure_contexts"]
            ):
                raise ValueError(
                    "Complete same physical evidence required; no fallback results used"
                )
            if slots == 3 and not report["control_passed"]:
                raise ValueError("Three-slot full3340 exact control required")
            reports[str(slots)] = report
            c.evaluate_round(rid)
            c.complete_round(rid)
            refresh()
        candidate = {m: reports["5"][m][VARIANT] for m in ("unknown", "known")}
        gate = development_gate(candidate, best["metrics"], c.config)
        movement(
            c.root / "rounds/0356_high_slot_depth_five_system",
            VARIANT,
            "results/research_loop/high_slot_depth_rank_movement.json",
        )
        write_json(
            "results/research_loop/high_slot_depth_attribution.json",
            {
                "local_only": True,
                "remote_publication": False,
                "independent_acceptance": False,
                "candidate_metrics": candidate,
                "gate_against_best236": gate,
                "control_round": "0355_high_slot_depth_three_control",
                "experimental_round": "0356_high_slot_depth_five_system",
                "control_all3340_full_ranks_exact": True,
                "scientific_best_changed": False,
                "scope": "Complete local cached repeated-development ranking experiment. Exact three-slot control and physical completion required; no automatic promotion or deployment. Any promising result still needs current-best/original206/historical181 gates and separate actual inference/resource verification.",
            },
        )
        refresh()


if __name__ == "__main__":
    main()
