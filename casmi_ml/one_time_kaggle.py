"""Publish exactly one explicit release while the research loop stays local-only."""

import argparse
import fcntl
import json
import time
from pathlib import Path

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.inference import validate_submission
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller, kernel_ref, now, release_identity
from casmi_ml.verified_best_release import VARIANT

BINDINGS = (
    "checkpoint_sha256",
    "baseline_sha256",
    "critic_checkpoint_sha256",
    "prefix",
    "slots",
    "open_protected",
    "frequency_weight",
    "token_length_exponent",
    "critic_weight",
    "adaptive_prefix",
    "expanded_prefix",
    "first_gate",
    "sampling",
    "formula_oracle_used",
)


def check_release(release, c):
    release = Path(release)
    sums = json.loads((release / "bundle/SHA256SUMS.json").read_text())
    if any(digest(release / "bundle" / name) != sha for name, sha in sums.items()):
        raise ValueError("Release content changed")
    identity = release_identity(release, sums, VARIANT)
    status = json.loads((release / "status.json").read_text())
    if status["identity"] != identity:
        raise ValueError("Release identity changed")
    proof = json.loads((release / "bundle/release_evidence.json").read_text())[
        "verified_inference"
    ]
    if proof["scientific_round"] != "0236_low_supported_leader_first" or not all(
        g["eligible"] for g in proof["gates"].values()
    ):
        raise ValueError("Frozen qualified236 scientific evidence required")
    verification = json.loads((release / "verification.json").read_text())
    if (
        not verification["valid"]
        or verification["identity"] != identity
        or verification["molecules"] != 400
    ):
        raise ValueError("Full400 local verification required")
    if (
        verification["seconds"] > c.config["inference_seconds"]
        or verification["peak_rss_mib"] > c.config["inference_rss_mib"]
    ):
        raise ValueError("Local resource gate failed")
    report = json.loads(
        (release / "local_output/submission.csv.report.json").read_text()
    )
    physical = report.get("external_routed", {}).get("status_counts", {})
    if any(n for s, n in physical.items() if "budget" in s):
        raise ValueError("Complete local physical evidence required")
    replay = json.loads((release / "implementation_replay.json").read_text())
    if (
        not replay["valid"]
        or replay["full_rank_matches"] != 75
        or not replay["fresh_fragment_cache"]
        or replay["development_budget_fallbacks"]
        or replay["bundle_sums_sha256"] != digest(release / "bundle/SHA256SUMS.json")
    ):
        raise ValueError("Source-bound complete fresh75 implementation replay required")
    validate_submission(
        pd.read_parquet("data/test.parquet"),
        pd.read_csv(release / "local_output/submission.csv"),
    )
    return identity


def authorize(release, c=None):
    c = c or Controller()
    if not c.local_only():
        raise ValueError("Persistent local-only research policy required")
    release = Path(release).resolve()
    identity = check_release(release, c)

    def operation(state):
        previous = state.get("one_time_kaggle")
        if previous:
            if previous["identity"] != identity:
                raise ValueError("One-time request already bound to another release")
            return previous
        value = {
            "identity": identity,
            "release": str(release),
            "status": "authorized",
            "authorized_at": now(),
            "maximum_accepted_submissions": 1,
            "github_pushes": False,
            "automatic_future_submissions": False,
            "authorization": "User: 现在先交一版本 给 kaggle 然后继续本地提升不需要提交",
        }
        state["one_time_kaggle"] = value
        return value

    return c.change(operation)


def amend_unsubmitted(release, c=None):
    """Repair the same requested science version without consuming another submission."""
    c = c or Controller()
    if not c.local_only():
        raise ValueError("Persistent local-only research policy required")
    release = Path(release).resolve()
    identity = check_release(release, c)
    previous = c.read()["one_time_kaggle"]
    old = Path(previous["release"])
    if (
        previous["status"] != "platform_verification_failed"
        or previous.get("submission_id") is not None
    ):
        raise ValueError(
            "Only a definitely unsubmitted platform failure may be amended"
        )
    old_sums, new_sums = [
        json.loads((p / "bundle/SHA256SUMS.json").read_text()) for p in (old, release)
    ]
    models = {name: sha for name, sha in old_sums.items() if Path(name).suffix == ".pt"}
    new_models = {
        name: sha for name, sha in new_sums.items() if Path(name).suffix == ".pt"
    }
    if not models or models != new_models:
        raise ValueError("Frozen model assets must remain identical")
    if digest(old / "bundle/deployment_recipe.json") != digest(
        release / "bundle/deployment_recipe.json"
    ):
        raise ValueError("Frozen scientific recipe must remain identical")
    old_rows, new_rows = [
        pd.read_csv(p / "local_output/submission.csv")
        .set_index("molecule_id")
        .sort_index()
        for p in (old, release)
    ]
    if len(old_rows) != 400 or not old_rows.equals(new_rows):
        raise ValueError("All400 original complete local rankings must remain exact")

    def operation(state):
        current = state["one_time_kaggle"]
        if (
            current != previous
            or state.get("pending_submission") is not None
            or current["identity"] in state["submissions"]
        ):
            raise ValueError("Accepted or ambiguous submission cannot be amended")
        updated = {
            key: current[key]
            for key in (
                "authorized_at",
                "authorization",
                "maximum_accepted_submissions",
                "github_pushes",
                "automatic_future_submissions",
            )
        }
        updated.update(
            identity=identity,
            release=str(release),
            status="authorized",
            amendments=[*current.get("amendments", []), current],
        )
        state["one_time_kaggle"] = updated
        return updated

    return c.change(operation)


def update(c, **fields):
    c.change(lambda s: s["one_time_kaggle"].update(fields, updated_at=now()))
    value = c.read()["one_time_kaggle"]
    receipt_path = Path(value["release"]) / "status.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        receipt.update(
            status=value["status"],
            uploaded=value.get("dataset_uploaded", False),
            submitted=value.get("submission_id") is not None,
            submission_id=value.get("submission_id"),
            public_score=value.get("public_score"),
            remote_release=value.get("remote"),
            automatic_future_submissions=False,
        )
        write_json(receipt_path, receipt)
    write_json(Path(value["release"]) / "one_time_submission.json", value)
    write_json(
        "results/research_loop/one_time_kaggle_submission.json",
        {k: v for k, v in value.items() if k != "release"},
    )
    return value


def verify_runtime_bindings(local, platform):
    """Bind optional runtime amendments without changing original science gates."""
    if "mass_reuse_runtime" in local:
        expected = local["mass_reuse_runtime"]
        if (
            not expected.get("original_deadlines_retained")
            or platform.get("mass_reuse_runtime") != expected
        ):
            raise ValueError("Platform mass-reuse runtime/source/classes differ")
    if "generation_cpu_runtime" in local:
        expected = local["generation_cpu_runtime"]
        if (
            expected.get("workers") != 2
            or expected.get("torch_cpu_threads") != 1
            or platform.get("generation_cpu_runtime") != expected
        ):
            raise ValueError("Platform CPU generator runtime/source differs")
    if "generation_physical_overlap" in local:
        expected = local["generation_physical_overlap"]
        actual = platform.get("generation_physical_overlap", {})
        for key in (
            "emitted_molecules",
            "stream_source_sha256",
            "overlap_source_sha256",
            "pipeline_source_sha256",
        ):
            if expected.get(key) != actual.get(key):
                raise ValueError("Platform generation overlap source/events differ")


def verify_physical_dictionaries(root):
    """New derived runtime requires complete exact platform physical evidence."""
    root = Path(root)
    results = [
        {
            p.name: json.loads(p.read_text())
            for p in (root / folder / "submission.csv.external_fragment_cache").glob(
                "*.json"
            )
        }
        for folder in ("local_output", "kaggle_output")
    ]
    local, platform = results
    if len(local) != 265 or len(platform) != 265 or local != platform:
        raise ValueError(
            "All265 original platform raw physical dictionaries must be exact"
        )
    if any(r.get("status") != "complete" for r in platform.values()):
        raise ValueError("Platform physical evidence incomplete")
    write_json(
        root / "platform_physical_dictionary_verification.json",
        {
            "valid": True,
            "complete_requests": 265,
            "all_original_raw_dictionaries_exact": True,
            "floating_tolerance_used": False,
            "independent_acceptance": False,
        },
    )


def verify_platform(release, remote, c):
    root = Path(release)
    local = (
        pd.read_csv(root / "local_output/submission.csv")
        .set_index("molecule_id")
        .sort_index()
    )
    platform = (
        pd.read_csv(root / "kaggle_output/submission.csv")
        .set_index("molecule_id")
        .sort_index()
    )
    validate_submission(pd.read_parquet("data/test.parquet"), platform.reset_index())
    if len(local) != 400 or not local.equals(platform):
        raise ValueError("All400 platform Top25 must match local verified release")
    a, b = [
        json.loads((root / folder / "submission.csv.report.json").read_text())
        for folder in ("local_output", "kaggle_output")
    ]
    verify_runtime_bindings(a, b)
    if any(a.get(k) != b.get(k) for k in BINDINGS):
        raise ValueError("Platform model or generation rule changed")
    for key in ("config", "source_sha256", "unlabeled_input"):
        if a["external_routed"].get(key) != b.get("external_routed", {}).get(key):
            raise ValueError("Platform external science/source differs")
    if any(
        n
        for status, n in b["external_routed"]["status_counts"].items()
        if "budget" in status
    ):
        raise ValueError("Platform physical evidence incomplete")
    if (
        b["seconds"] > c.config["inference_seconds"]
        or b["peak_rss_mib"] > c.config["inference_rss_mib"]
    ):
        raise ValueError("Platform resource gate failed")
    if "mass_reuse_runtime" in a:
        verify_physical_dictionaries(root)
    result = {
        "valid": True,
        "identity": check_release(root, c),
        "molecules": 400,
        "local_full_top25_matches": 400,
        "seconds": b["seconds"],
        "parent_peak_rss_mib": b["peak_rss_mib"],
        "remote_release": remote,
        "independent_acceptance": False,
    }
    write_json(root / "kaggle_verification.json", result)
    return result


def reserve(c, value):
    def operation(state):
        one = state["one_time_kaggle"]
        if (
            not state.get("local_only")
            or one["identity"] != value["identity"]
            or one["status"] not in ("platform_verified", "quota_wait")
        ):
            raise ValueError("Unused explicit single-release authorization required")
        if state.get("pending_submission") is not None:
            return False
        if state.get("submit_after") and now() < state["submit_after"]:
            return False
        if one.get("submission_id") is not None:
            raise ValueError("Single submission authorization already consumed")
        one.update(status="submission_intent", submission_intent_at=now())
        state["pending_submission"] = one["identity"]
        state["submissions"][one["identity"]] = {
            "identity": one["identity"],
            "version": one["remote"]["version"],
            "message": one["message"],
            "status": "intent",
            "created_at": now(),
            "id": None,
        }
        return True

    return c.change(operation)


def step(c, api):
    if c.stopped():
        return "stopped"
    value = c.read()["one_time_kaggle"]
    if not c.local_only():
        raise ValueError("One-time publisher never enables general remote publication")
    release = Path(value["release"])
    if check_release(release, c) != value["identity"]:
        raise ValueError("One-time authorization release changed")
    status = value["status"]
    if status in (
        "COMPLETE",
        "ERROR",
        "CANCELLED",
        "failed",
        "upload_intent",
        "kernel_intent",
    ):
        return status
    if status in ("submission_intent", "PENDING"):
        rows = api.competition_submissions(c.config["competition"])
        matches = [r for r in rows if r.description == value["message"]]
        if len(matches) > 1:
            raise ValueError("Ambiguous duplicate remote submissions")
        if not matches:
            return "submission_intent"
        row = matches[0]
        if value.get("submission_id") and int(row.ref) != value["submission_id"]:
            raise ValueError("Remote submission identifier changed")
        c.finish_submission(value["identity"], row.ref)
        remote_status = str(row.status).split(".")[-1].upper()
        score = float(row.public_score) if row.public_score not in (None, "") else None
        c.update_submission(value["identity"], remote_status, score)
        update(c, status=remote_status, submission_id=int(row.ref), public_score=score)
        return remote_status
    if not value.get("dataset_uploaded"):
        update(c, status="upload_intent")
        response = api.dataset_create_new(
            str(release / "dataset"), public=False, convert_to_csv=False, quiet=True
        )
        if response.error or str(response.status).lower() == "error":
            raise ValueError(
                "Kaggle rejected private release dataset: " + str(response)
            )
        update(c, dataset_uploaded=True, status="dataset_uploaded")
        return "dataset_uploaded"
    if not value.get("remote"):
        dataset_id = json.loads(
            (release / "dataset/dataset-metadata.json").read_text()
        )["id"]
        if api.dataset_status(dataset_id) != "ready":
            return "dataset_processing"
        update(c, status="kernel_intent")
        response = api.kernels_push(str(release / "notebook"), timeout="1800")
        if response.error and "Maximum batch GPU session count" in response.error:
            if response.ref or response.version_number not in (None, 0):
                raise ValueError("Ambiguous kernel capacity rejection")
            update(c, status="dataset_uploaded", capacity_wait=response.error)
            return "kernel_capacity_wait"
        if (
            response.error
            or response.invalid_dataset_sources
            or response.invalid_competition_sources
        ):
            raise ValueError("Kaggle rejected notebook: " + str(response))
        update(
            c,
            remote={
                "kernel": kernel_ref(response.ref),
                "version": int(response.version_number),
            },
            status="notebook_running",
        )
        return "notebook_running"
    if status == "notebook_running":
        response = api.kernels_status(value["remote"]["kernel"])
        remote_status = (
            str(response["status"] if isinstance(response, dict) else response.status)
            .split(".")[-1]
            .upper()
        )
        if remote_status == "ERROR":
            update(
                c,
                status="failed",
                error="Kaggle notebook failed; no competition submission sent",
            )
            return "failed"
        if remote_status != "COMPLETE":
            return "notebook_running"
        pattern = r"(^|/)(submission\.csv(?:\.report\.json|\.external\.report\.json)?|retrieval\.csv\.report\.json)$"
        local_report = json.loads(
            (release / "local_output/submission.csv.report.json").read_text()
        )
        if "mass_reuse_runtime" in local_report:
            pattern += (
                r"|(^|/)submission\.csv\.external_fragment_cache/[^/]+\.threads2\.json$"
            )
        api.kernels_output(
            value["remote"]["kernel"],
            str(release / "kaggle_output"),
            file_pattern=pattern,
            force=True,
            quiet=True,
            page_size=100,
        )
        verify_platform(release, value["remote"], c)
        message = (
            "One requested CASMI scientific236 verified best "
            + value["identity"][:16]
            + "; repeated development unknownMRR .043978301, known .603792116; no independent accuracy claim."
        )
        update(c, status="platform_verified", message=message)
        return "platform_verified"
    if status in ("platform_verified", "quota_wait"):
        platform = json.loads((release / "kaggle_verification.json").read_text())
        if (
            not platform["valid"]
            or platform["identity"] != value["identity"]
            or platform["local_full_top25_matches"] != 400
        ):
            raise ValueError("Bound complete platform verification required")
        if not reserve(c, value):
            update(c, status="quota_wait", retry_at=c.read().get("submit_after"))
            return "quota_wait"
        try:
            response = api.competition_submit_code(
                "submission.csv",
                value["message"],
                competition=c.config["competition"],
                kernel=value["remote"]["kernel"],
                kernel_version=value["remote"]["version"],
            )
        except Exception as error:
            from requests import HTTPError

            if (
                isinstance(error, HTTPError)
                and error.response.status_code == 400
                and "daily Submission allowance" in error.response.text
            ):
                c.defer_submission(
                    value["identity"],
                    "Kaggle rejected before acceptance: daily submission allowance exhausted",
                )
                update(c, status="quota_wait", retry_at=c.read()["submit_after"])
                return "quota_wait"
            # An uncertain accepted POST is reconciled read-only, never resent.
            update(c, error=str(error))
            raise
        c.finish_submission(value["identity"], response.ref)
        update(c, status="PENDING", submission_id=int(response.ref))
        return "PENDING"
    raise ValueError("Unknown one-time publisher state: " + status)


def main():
    from kaggle.api.kaggle_api_extended import KaggleApi

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--release", type=Path)
    p.add_argument("--authorize", action="store_true")
    p.add_argument("--amend-unsubmitted", action="store_true")
    p.add_argument("--loop", action="store_true")
    args = p.parse_args()
    c = Controller()
    if args.authorize:
        if args.release is None:
            p.error("--release required for explicit authorization")
        authorize(args.release, c)
    if args.amend_unsubmitted:
        if args.release is None:
            p.error("--release required for runtime amendment")
        amend_unsubmitted(args.release, c)
    api = KaggleApi()
    api.authenticate()
    with (c.root / "one_time_kaggle.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        while True:
            try:
                status = step(c, api)
            except Exception as error:
                print(type(error).__name__, str(error), flush=True)
                current = c.read()["one_time_kaggle"]["status"]
                if current in ("upload_intent", "kernel_intent") or isinstance(
                    error, ValueError
                ):
                    update(c, error=str(error))
                    raise
                status = current
            print(now(), status, flush=True)
            if not args.loop or status in (
                "COMPLETE",
                "ERROR",
                "CANCELLED",
                "failed",
                "stopped",
            ):
                return
            time.sleep(45)


if __name__ == "__main__":
    main()
