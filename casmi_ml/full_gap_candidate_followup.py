"""Local candidate-specific diagnostic chemistry cycle with full zero control first."""

import fcntl
import json

from casmi_ml.completed_system_reports import wait_for_complete_system
from casmi_ml.data import write_json
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.full_gap_candidate_compatibility import BONUS_BUDGET, DIM
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, TRAIN, freeze


def prepare_zero(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "zero_model.json"
    value = {
        "weights": [0.0] * DIM,
        "mask_chemical_channels": False,
        "bonus_budget": BONUS_BUDGET,
        "encoder_sha256": digest(ENCODER),
        "original_critic_sha256": digest(CRITIC),
        "compatibility_source_sha256": digest(
            "casmi_ml/full_gap_candidate_compatibility.py"
        ),
        "chemistry_source_sha256": digest("casmi_ml/chemistry.py"),
        "extended_rule_source_sha256": digest("casmi_ml/extended_diagnostic_rules.py"),
        "evidence_source_sha256": digest("casmi_ml/diagnostic_residual.py"),
        "feature_source_sha256": digest("casmi_ml/diagnostic_feature_runtime.py"),
        "training_sha256": digest(TRAIN),
        "local_only": True,
        "independent_acceptance": False,
    }
    freeze(path, value)
    return path


def complete_system(c, model, rid, control=False):
    for attempt in range(3):
        if c.stopped():
            return None
        trial = rid if attempt == 0 else rid + f"_cache_completion{attempt}"
        source, decision = execute(
            c,
            trial,
            "casmi_ml.full_gap_candidate_system",
            ["--model", str(model), *(["--control"] if control else [])],
            "implementation_control"
            if control
            else "candidate_chemistry_compatibility",
        )
        report = json.loads((source / "report.json").read_text())
        if (
            not report["development_budget_fallbacks"]
            and not report["incremental_new_failure_contexts"]
        ):
            return source, decision
    raise RuntimeError(
        "Bounded extended candidate chemistry physical completion exhausted: " + rid
    )


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError(
            "Candidate chemistry cycle requires persistent local-only mode"
        )
    with (c.root / "full_gap_candidate_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if wait_for_complete_system(c, "0324_max_candidate_mask_system") is None:
            return
        freeze(
            c.root / "full_gap_candidate_preregistration.json",
            {
                "hypothesis": "Training-only325 fixed full cosine-gap capacity2 audit improves original mean-pool objective.5290418->.5246472, learned sum1.715<2 and max.262. Original.05 cap binds; test training-L2-regularized chemistry under fixed theoretical2 ceiling; contrast with changing query vectors, no absent-peak penalty.",
                "training": "Original60K train only, original frozen encoder/critic, original pinned unrestricted15 negatives, same RNG42 representative/RNG260 same-ion pair as260/261. Convex60D nonnegative weights, total2.0 full theoretical normalized-cosine gap ceiling, CE frozen scale and L2.01, fixedSLSQP200 ftol1e-10. No dev-selected weights or budget.",
                "difference_from_0049": "Separate60 rule/kind channels from original8+12 broad hypotheses, frozen original critic offset. Only candidate-chemical hypothesis weight budget.05->2.0 changes versus307/308. Same20 rules/60 channels, original mean pooling/raw sqrt strength restored, no max/squared combination, same train-onlyL2.01 and frozen critic scale.2.0 is mathematical cosine full difference, not a dev-selected multiplier. Deployment margin remains.05; all science gates unchanged.",
                "controls": "First zero bonus complete3340 exact ranking comparison to237; matched identity-masked training then full3340 original-critic control. No parameter tuning based on chemical dev outcome.",
                "promotion": "Existing frozen gates versus current best, completed original0206 and historical181. No promotion on incomplete physical evidence/budget fallback. Winner requires explicit runtime binding before activation.",
                "representation_seconds_per_arm": 86400,
                "fragment_seconds_per_scenario": 1200,
                "audit_report_sha256": digest(
                    c.root
                    / "rounds/0325_training_candidate_bonus_capacity_audit/report.json"
                ),
                "source_sha256": digest(__file__),
                "training_source_sha256": digest(
                    "casmi_ml/full_gap_candidate_training.py"
                ),
                "score_source_sha256": digest(
                    "casmi_ml/full_gap_candidate_compatibility.py"
                ),
                "system_source_sha256": digest("casmi_ml/full_gap_candidate_system.py"),
                "local_only": True,
                "independent_acceptance": False,
            },
        )
        zero = prepare_zero(c.root / "rounds/0326_full_gap_candidate_zero_control")
        completed = complete_system(
            c, zero, "0326_full_gap_candidate_zero_control", True
        )
        if completed is None:
            return
        source, _ = completed
        if not json.loads((source / "report.json").read_text())["control_passed"]:
            raise RuntimeError("Candidate chemistry full zero control failed")
        models = []
        for mask, rid in [
            (False, "0327_full_gap_candidate_chemical"),
            (True, "0328_full_gap_candidate_mask"),
        ]:
            if c.stopped():
                return
            out = c.root / "rounds" / rid
            c.register_round(
                rid,
                "representation_diagnostic",
                [
                    ".venv-gpu/bin/python",
                    "-u",
                    "-m",
                    "casmi_ml.full_gap_candidate_training",
                    "--output",
                    str(out),
                    *(["--mask-chemical-channels"] if mask else []),
                ],
                out / "report.json",
            )
            with (c.root / "gpu.lock").open("a") as gpu_lock:
                fcntl.flock(gpu_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                c.run_auxiliary(rid)
            c.evaluate_round(rid)
            c.complete_round(rid)
            models.append(out / "model.json")
        arms = {}
        for name, model, rid in [
            ("chemical", models[0], "0329_full_gap_candidate_chemical_system"),
            ("identity_masked", models[1], "0330_full_gap_candidate_mask_system"),
        ]:
            completed = complete_system(c, model, rid, name == "identity_masked")
            if completed is None:
                return
            source, decision = completed
            report = json.loads((source / "report.json").read_text())
            arms[name] = {
                "round": source.name,
                "metrics": {
                    m: report[m]["full_gap_candidate_compatibility"]
                    for m in ("unknown", "known")
                },
                "winner": decision.get("winner"),
                "runtime_activation_required": bool(decision.get("winner")),
            }
        write_json(
            "results/research_loop/full_gap_candidate_attribution.json",
            {
                "local_only": True,
                "remote_publication": False,
                "independent_acceptance": False,
                "arms": arms,
                "chemical_minus_mask_unknown_mrr25": arms["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - arms["identity_masked"]["metrics"]["unknown"]["mrr25"],
                "scope": "Complete repeated development3340 with bounded candidate SMARTS-conditioned observed chemistry, unchanged original encoder/critic and physical routing. Mask equals original critic; not independent acceptance or cold inference.",
            },
        )


if __name__ == "__main__":
    main()
