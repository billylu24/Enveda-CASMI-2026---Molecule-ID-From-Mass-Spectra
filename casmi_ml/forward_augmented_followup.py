"""Full local complementary diagnostic support cycle; finalize only after both exact controls."""

import fcntl
import json
import time

from casmi_ml.data import write_json
from casmi_ml.forward_augmented_system import VARIANT
from casmi_ml.full_gap_candidate_followup import prepare_zero
from casmi_ml.local_research_snapshot import refresh
from casmi_ml.metfrag import digest
from casmi_ml.query_cycle_aggregate import refresh as aggregate
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import freeze


def bound_zero(output):
    from casmi_ml.forward_augmented_training import AUDIT, bind_model

    path = prepare_zero(output)
    model = bind_model(
        json.loads(path.read_text()),
        AUDIT / "private_forward_augmented_signature_model.npz",
    )
    target = output / "forward_augmented_zero_model.json"
    freeze(target, model)
    return target


def complete(c, model, rid, control=False):
    for attempt in range(3):
        if c.stopped():
            return None
        trial = rid if attempt == 0 else rid + f"_cache_completion{attempt}"
        out = c.root / "rounds" / trial
        c.register_round(
            trial,
            "candidate_chemistry_compatibility"
            if not control
            else "implementation_control",
            [
                ".venv/bin/python",
                "-u",
                "-m",
                "casmi_ml.forward_augmented_system",
                "--output",
                str(out),
                "--model",
                str(model),
                *(["--control"] if control else []),
            ],
            out / "report.json",
        )
        c.run_auxiliary(trial)
        report = json.loads((out / "report.json").read_text())
        if (
            not report["development_budget_fallbacks"]
            and not report["incremental_new_failure_contexts"]
        ):
            if control and not report["control_passed"]:
                raise ValueError("Forward-augmented zero/mask exact control failed")
            # Require original IDs for the source-bound activation controls.
            if control and trial != rid:
                raise ValueError(
                    "Control physical completion needs explicit new proof binding"
                )
            return out
        c.evaluate_round(trial)
        c.complete_round(trial)
    raise ValueError("Forward-augmented physical completion exhausted")


def main():
    c = Controller()
    if not c.local_only():
        raise ValueError("Forward-augmented cycle is local-only")
    with (c.root / "forward_augmented_followup.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        probe = json.loads(
            (
                c.root / "rounds/0374_forward_augmented_capacity_audit/report.json"
            ).read_text()
        )
        audit_protocol = json.loads(
            (
                c.root
                / "rounds/0374_forward_augmented_capacity_audit/forward_augmented_capacity_protocol.json"
            ).read_text()
        )
        for field, source in {
            "source_sha256": "casmi_ml/forward_augmented_capacity_audit.py",
            "base_training_source_sha256": "casmi_ml/full_gap_candidate_training.py",
        }.items():
            if audit_protocol[field] != digest(source):
                raise ValueError("Training capacity evidence source changed: " + field)
        if probe["capacity_probe"]["same_objective_gain"] <= 0:
            raise ValueError(
                "No matched training objective gain; do not run development variants"
            )
        freeze(
            c.root / "forward_augmented_preregistration.json",
            {
                "source_sha256": digest(__file__),
                "audit_sha256": digest(
                    c.root / "rounds/0374_forward_augmented_capacity_audit/report.json"
                ),
                "hypothesis": "372 fixed train-internal heldout fingerprint ridge predicts mean measured60D better than20SMARTS by8.8%MSE.373 direct support replacement fails capacity.374 complementary original60binary plus60 predicted diagnostic supports gives objective.524647178->.523612107, gain.001035071 under same total2/L2.01. Original model is embedded feasible zero-extra solution; training gain is not ranking generalization. Test fixed120D original60K/120K/pinned15/unrestricted denominator/critic/encoder/mean/sqrt/margin.05. No mass-mask or threshold combination, no development parameter selection.",
                "protocol": "Zero complete3340 exact before original60K paired chemical/mask fit. Both complete physical3340 chemical and exact mask BEFORE scientific evaluation. Freeze best236/original0206/historical181 gates and admissionmargin.05. No new holdout, no remote publication.",
                "representation_seconds_per_arm": 86400,
                "fragment_seconds_per_scenario": 1200,
                "local_only": True,
                "independent_acceptance": False,
            },
        )
        zero = bound_zero(c.root / "rounds/0377_forward_augmented_zero_control")
        directory = complete(c, zero, "0377_forward_augmented_zero_control", True)
        if directory is None:
            return
        c.evaluate_round(directory.name)
        c.complete_round(directory.name)
        models = []
        for mask, rid in [
            (False, "0378_forward_augmented_chemical"),
            (True, "0379_forward_augmented_mask"),
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
                    "casmi_ml.forward_augmented_training",
                    "--output",
                    str(out),
                    *(["--mask-chemical-channels"] if mask else []),
                ],
                out / "report.json",
            )
            while not c.stopped():
                with (c.root / "gpu.lock").open("a") as gpu:
                    try:
                        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        time.sleep(45)
                        continue
                    c.run_auxiliary(rid)
                    break
            if c.stopped():
                return
            c.evaluate_round(rid)
            c.complete_round(rid)
            models.append(out / "model.json")
        chemical = complete(c, models[0], "0380_forward_augmented_chemical_system")
        if chemical is None:
            return
        mask = complete(c, models[1], "0381_forward_augmented_mask_system", True)
        if mask is None:
            return
        c.evaluate_round(mask.name)
        c.complete_round(mask.name)
        decision = c.evaluate_round(chemical.name)
        c.complete_round(chemical.name)
        # Runtime activation requires a separate source-bound actual-entry proof.
        # This driver completes paired scientific evidence only.
        arms = {}
        for name, directory in [("chemical", chemical), ("identity_masked", mask)]:
            report = json.loads((directory / "report.json").read_text())
            arms[name] = {
                "round": directory.name,
                "metrics": {
                    mode: report[mode][VARIANT] for mode in ("unknown", "known")
                },
            }
        write_json(
            "results/research_loop/forward_augmented_attribution.json",
            {
                "local_only": True,
                "remote_publication": False,
                "independent_acceptance": False,
                "arms": arms,
                "chemical_minus_mask_unknown_mrr25": arms["chemical"]["metrics"][
                    "unknown"
                ]["mrr25"]
                - arms["identity_masked"]["metrics"]["unknown"]["mrr25"],
                "winner": bool(decision.get("winner")),
                "scope": "Complete repeated development fixed120D original binary+predicted measured diagnostic candidate support, matched exact zero/mask controls. Extra feature capacity training gain is not independent accuracy evidence; no remote publication or runtime activation.",
            },
        )
        refresh()
        aggregate()
        print(
            "Finalized forward-augmented cycle winner",
            bool(decision.get("winner")),
            flush=True,
        )


if __name__ == "__main__":
    main()
