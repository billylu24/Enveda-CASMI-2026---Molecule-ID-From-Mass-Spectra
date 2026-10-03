"""Local candidate-specific diagnostic chemistry cycle with full zero control first."""

import fcntl
import json

from casmi_ml.candidate_diagnostic_compatibility import BONUS_BUDGET
from casmi_ml.data import write_json
from casmi_ml.diagnostic_local_followup import execute
from casmi_ml.generated_first_critic import CRITIC
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ENCODER, TRAIN, freeze


def prepare_zero(output):
    output.mkdir(parents=True, exist_ok=True)
    path = output / "zero_model.json"
    value = {
        "weights": [0.0] * 24,
        "mask_chemical_channels": False,
        "bonus_budget": BONUS_BUDGET,
        "encoder_sha256": digest(ENCODER),
        "original_critic_sha256": digest(CRITIC),
        "compatibility_source_sha256": digest(
            "casmi_ml/candidate_diagnostic_compatibility.py"
        ),
        "chemistry_source_sha256": digest("casmi_ml/chemistry.py"),
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
            "casmi_ml.candidate_compatibility_system",
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
        "Bounded candidate chemistry physical completion exhausted: " + rid
    )


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError(
            "Candidate chemistry cycle requires persistent local-only mode"
        )
    with (c.root / "candidate_compatibility_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        freeze(
            c.root / "candidate_compatibility_preregistration.json",
            {
                "hypothesis": "Observed diagnostic-ion and neutral-loss identities interact directly with actual candidate SMARTS motifs; contrast with changing query vectors, no absent-peak penalty.",
                "training": "Original60K train only, original frozen encoder/critic, original pinned unrestricted15 negatives, same RNG42 representative/RNG260 same-ion pair as260/261. Convex24D nonnegative weights, total .05 cosine cap, CE frozen scale and L2.01, fixedSLSQP200 ftol1e-10. No dev-selected weights or budget.",
                "difference_from_0049": "Separate24 rule/kind interactions, frozen deployed critic cosine offset, full actual external candidate paths.0049 used generated pools/independent pairwise ranking and8 merged rule scores.",
                "controls": "First zero bonus complete3340 exact ranking comparison to237; matched identity-masked training then full3340 original-critic control. No parameter tuning based on chemical dev outcome.",
                "promotion": "Existing frozen gates versus current best, completed original0206 and historical181. No promotion on incomplete physical evidence/budget fallback. Winner requires explicit runtime binding before activation.",
                "representation_seconds_per_arm": 86400,
                "fragment_seconds_per_scenario": 1200,
                "source_sha256": digest(__file__),
                "training_source_sha256": digest(
                    "casmi_ml/candidate_compatibility_training.py"
                ),
                "score_source_sha256": digest(
                    "casmi_ml/candidate_diagnostic_compatibility.py"
                ),
                "system_source_sha256": digest(
                    "casmi_ml/candidate_compatibility_system.py"
                ),
                "local_only": True,
                "independent_acceptance": False,
            },
        )
        zero = prepare_zero(c.root / "rounds/0299_candidate_compatibility_zero_control")
        completed = complete_system(
            c, zero, "0299_candidate_compatibility_zero_control", True
        )
        if completed is None:
            return
        source, _ = completed
        if not json.loads((source / "report.json").read_text())["control_passed"]:
            raise RuntimeError("Candidate chemistry full zero control failed")
        models = []
        for mask, rid in [
            (False, "0300_candidate_compatibility_chemical"),
            (True, "0301_candidate_compatibility_mask"),
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
                    "casmi_ml.candidate_compatibility_training",
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
            ("chemical", models[0], "0302_candidate_compatibility_chemical_system"),
            ("identity_masked", models[1], "0303_candidate_compatibility_mask_system"),
        ]:
            completed = complete_system(c, model, rid, name == "identity_masked")
            if completed is None:
                return
            source, decision = completed
            report = json.loads((source / "report.json").read_text())
            arms[name] = {
                "round": source.name,
                "metrics": {
                    m: report[m]["candidate_compatibility"]
                    for m in ("unknown", "known")
                },
                "winner": decision.get("winner"),
                "runtime_activation_required": bool(decision.get("winner")),
            }
        write_json(
            "results/research_loop/candidate_compatibility_attribution.json",
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
