"""Durable research control, explicit development gates, and bounded child jobs."""

import argparse
import fcntl
import hashlib
import json
import math
import os
import signal
import subprocess
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze

CONFIG = Path("configs/research_loop.json")


def now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def development_gate(new, baseline, config):
    p = config["development"]
    n, b = new["unknown"], baseline["unknown"]
    absolute = n["mrr25"] - b["mrr25"]
    relative = (
        absolute / b["mrr25"] if b["mrr25"] > 0 else (1.0 if absolute > 0 else 0.0)
    )
    finite = all(
        math.isfinite(v)
        for v in [
            absolute,
            relative,
            new["known"]["mrr25"],
            new["known"]["top1"],
            baseline["known"]["mrr25"],
            baseline["known"]["top1"],
        ]
    )
    reasons = []
    if any(
        new[m]["molecules"] != baseline[m]["molecules"] for m in ["known", "unknown"]
    ):
        reasons.append("cohort_size_mismatch")
    if not finite:
        reasons.append("nonfinite_metrics")
    for report in [new, baseline]:
        for mode in ["unknown", "known"]:
            if any(
                not math.isfinite(report[mode][k]) or not 0 <= report[mode][k] <= 1
                for k in ["mrr25", "top1"]
            ):
                reasons.append("invalid_metric_range")
            if (
                not isinstance(report[mode]["molecules"], int)
                or report[mode]["molecules"] < 1
            ):
                reasons.append("invalid_molecule_count")
    if min(n["molecules"], b["molecules"]) < p["minimum_molecules"]:
        reasons.append("insufficient_development_molecules")
    if absolute + 1e-12 < p["minimum_absolute_gain"]:
        reasons.append("absolute_gain_below_gate")
    if relative + 1e-12 < p["minimum_relative_gain"]:
        reasons.append("relative_gain_below_gate")
    if (
        new["known"]["mrr25"] + 1e-12
        < baseline["known"]["mrr25"] - p["known_mrr_tolerance"]
    ):
        reasons.append("known_mrr_protection_failed")
    if (
        new["known"]["top1"] + 1e-12
        < baseline["known"]["top1"] - p["known_top1_tolerance"]
    ):
        reasons.append("known_top1_protection_failed")
    if min(new["known"]["molecules"], baseline["known"]["molecules"]) < 1:
        reasons.append("no_known_evaluation")
    return {
        "eligible": not reasons,
        "absolute_gain": absolute,
        "relative_gain": relative,
        "reasons": reasons,
        "evidence": "development_experimental; not independent acceptance",
    }


def kernel_ref(value):
    return str(value).removeprefix("https://www.kaggle.com").removeprefix("/code/")


def fold_for(key, folds=5):
    return (
        int.from_bytes(
            hashlib.sha256(("research-loop:" + str(key)).encode()).digest()[:8], "big"
        )
        % folds
    )


def fold_summary(selected, baseline, folds=5):
    a = selected.set_index("key")
    b = baseline.set_index("key")
    if (
        a.index.duplicated().any()
        or b.index.duplicated().any()
        or set(a.index) != set(b.index)
    ):
        raise ValueError("Paired development keys differ")
    rows = []
    for fold in range(folds):
        ids = [key for key in a.index if fold_for(key, folds) == fold]
        if not ids:
            raise ValueError("Empty development fold")
        rows.append(
            {
                "fold": fold,
                "molecules": len(ids),
                "baseline_mrr25": float(b.loc[ids].reciprocal_rank.mean()),
                "mrr25": float(a.loc[ids].reciprocal_rank.mean()),
                "difference": float(
                    (a.loc[ids].reciprocal_rank - b.loc[ids].reciprocal_rank).mean()
                ),
            }
        )
    return {
        "folds": rows,
        "label": "Fixed molecule folds, aggregate development diagnostics; repeated use is not fresh independent validation",
    }


def content_identity(files, config):
    files = sorted(map(Path, files), key=str)
    if not files:
        raise ValueError("Identity requires release files")
    common = Path(os.path.commonpath([str(p.parent.resolve()) for p in files]))
    contents = {str(p.resolve().relative_to(common)): digest(p) for p in files}
    return hashlib.sha256(
        json.dumps({"files": contents, "config": config}, sort_keys=True).encode()
    ).hexdigest()


def release_identity(release, sums, variant):
    release = Path(release)
    files = [release / "bundle" / name for name in sums]
    notebook = release / "notebook"
    if notebook.exists():
        files.extend(notebook.glob("*.ipynb"))
        files.extend(notebook.glob("kernel-metadata.json"))
    return content_identity(files, {"variant": variant})


class Controller:
    def __init__(self, config=CONFIG):
        self.config_path = Path(config)
        self.config = json.loads(self.config_path.read_text())
        self.root = Path(self.config["root"])
        self.root.mkdir(parents=True, exist_ok=True)
        freeze(self.root / "config.json", self.config)
        self.path = self.root / "state.json"
        self.stop_path = self.root / "STOP"
        if not self.path.exists():
            write_json(
                self.path,
                {
                    "version": 1,
                    "status": "ready",
                    "created_at": now(),
                    "updated_at": now(),
                    "rounds": [],
                    "submissions": {},
                    "pending_submission": None,
                    "development_best": None,
                    "public_best": {"score": 0.176, "submission_id": 56535378},
                    "active_job": None,
                    "github_commits": [],
                },
            )

    def read(self):
        return json.loads(self.path.read_text())

    def change(self, operation):
        with (self.root / "state.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.read()
            result = operation(state)
            state["updated_at"] = now()
            write_json(self.path, state)
            return result

    def stopped(self):
        return self.stop_path.exists()

    def stop(self):
        self.stop_path.write_text(now() + "\n")
        self.change(lambda s: s.update(status="stop_requested"))
        state = self.read()
        entries = list(state.get("auxiliary_jobs", {}).values())
        if state.get("active_job"):
            entries.append(state["active_job"])
        entries += [
            {"pid": r["external_pid"], "argv": r["argv"]}
            for r in state["rounds"]
            if r.get("external_pid") and r["status"] == "external_running"
        ]
        for entry in entries:
            proc = Path(f"/proc/{entry['pid']}/cmdline")
            try:
                actual = [x.decode() for x in proc.read_bytes().split(b"\0") if x]
                if actual == entry["argv"]:
                    os.killpg(entry["pid"], signal.SIGTERM)
            except ProcessLookupError:
                pass
            except FileNotFoundError:
                pass

    def resume(self):
        self.stop_path.unlink(missing_ok=True)
        self.change(lambda s: s.update(status="ready"))

    def register_round(self, identifier, direction, argv, report):
        def update(s):
            if any(r["id"] == identifier for r in s["rounds"]):
                return
            s["rounds"].append(
                {
                    "id": identifier,
                    "direction": direction,
                    "argv": argv,
                    "report": str(report),
                    "status": "queued",
                    "created_at": now(),
                    "git_synced": False,
                    "identity": None,
                }
            )

        self.change(update)

    def job(self, argv, log, seconds=None, auxiliary_key=None):
        if self.stopped():
            return "stopped"
        log = Path(log)
        log.parent.mkdir(parents=True, exist_ok=True)
        budget = None
        if seconds is not None:
            from casmi_ml.research_budget import StageBudget

            # Each purpose freezes its own allowance: an experiment, replay and
            # verification can share a directory but have different deadlines.
            stage = "wall_job:" + log.name
            ledger_path = log.parent / "training_budget.json"
            if ledger_path.exists():
                legacy = json.loads(ledger_path.read_text()).get("wall_job", {})
                if str(log) in legacy.get("runs", {}):
                    stage = "wall_job"
            budget = StageBudget(
                log.parent,
                stage,
                str(log),
                seconds,
                limit=seconds,
                lock_path=log.parent / "job.lock",
            )
            seconds = budget.allowance
        start = time.monotonic()
        with log.open("a") as stream:
            child = subprocess.Popen(
                argv, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
            )
            entry = {"pid": child.pid, "argv": argv, "started_at": now()}

            def record(s):
                if auxiliary_key is None:
                    s["active_job"] = entry
                else:
                    s.setdefault("auxiliary_jobs", {})[auxiliary_key] = entry
                s["status"] = "running"

            self.change(record)
            try:
                while child.poll() is None:
                    if self.stopped() or (
                        seconds is not None and time.monotonic() - start >= seconds
                    ):
                        try:
                            os.killpg(child.pid, signal.SIGTERM)
                        except ProcessLookupError:
                            pass
                        try:
                            child.wait(timeout=10)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                        return "stopped" if self.stopped() else "budget_exhausted"
                    time.sleep(0.2)
                if self.stopped():
                    return "stopped"
                if child.returncode:
                    raise RuntimeError(
                        f"Child exited {child.returncode}: {argv}; see {log}"
                    )
            finally:
                if budget is not None:
                    budget.close()
                if auxiliary_key is None:
                    self.change(lambda s: s.update(active_job=None))
                else:
                    self.change(
                        lambda s: s.setdefault("auxiliary_jobs", {}).pop(
                            auxiliary_key, None
                        )
                    )
        return "complete"

    def run_auxiliary(self, identifier):
        state = self.read()
        r = next(r for r in state["rounds"] if r["id"] == identifier)
        entries = [state.get("active_job"), *state.get("auxiliary_jobs", {}).values()]
        for entry in entries:
            if not entry or entry["argv"] != r["argv"]:
                continue
            try:
                actual = [
                    x.decode()
                    for x in Path(f"/proc/{entry['pid']}/cmdline")
                    .read_bytes()
                    .split(b"\0")
                    if x
                ]
            except FileNotFoundError:
                continue
            if actual == r["argv"]:
                return "orphan_job_running"
        path = Path(r["report"])
        if path.exists():
            self.mark_round(identifier, status="evaluated")
            return "evaluated"
        self.mark_round(identifier, status="external_running")
        try:
            result = self.job(
                r["argv"],
                path.parent / "execution.log",
                self.config["gpu_stage_seconds"]["generation"],
                auxiliary_key=identifier,
            )
            if result == "complete" and not path.exists():
                raise ValueError("Auxiliary experiment exited without report")
            self.mark_round(
                identifier,
                status="evaluated" if result == "complete" else "retry",
                completed_at=now(),
            )
            return result
        except Exception as error:
            status = "retry" if self.stopped() else "failed"
            self.mark_round(
                identifier, status=status, error=str(error), completed_at=now()
            )
            raise

    def reserve_submission(self, identity, version, message):
        def update(s):
            if self.stopped():
                raise RuntimeError("Stop requested")
            if identity in s.get("submission_aliases", {}):
                raise RuntimeError(
                    "This content identity aliases an existing submission"
                )
            if s["pending_submission"] is not None:
                raise RuntimeError("One submission is already pending")
            previous = s["submissions"].get(identity)
            if previous is not None and previous["status"] != "quota_wait":
                raise RuntimeError("This content was already submitted or reserved")
            if s.get("submit_after") and now() < s["submit_after"]:
                raise RuntimeError("Daily Kaggle allowance has not reset")
            s["submissions"][identity] = {
                "identity": identity,
                "version": version,
                "message": message,
                "status": "intent",
                "created_at": now(),
                "id": None,
                "rejections": previous.get("rejections", []) if previous else [],
            }
            s["pending_submission"] = identity

        self.change(update)

    def finish_submission(self, identity, submission_id):
        self.change(
            lambda s: s["submissions"][identity].update(
                id=int(submission_id), status="PENDING"
            )
        )

    def publication_snapshot(self, identity):
        state = self.read()
        entry = state["submissions"][identity]
        for row in state["rounds"]:
            if row.get("identity") != identity:
                continue
            snapshot = {
                key: entry.get(key)
                for key in ["id", "status", "score", "version", "error"]
            }
            if row.get("release"):
                path = Path(row["release"]) / "status.json"
                if path.exists():
                    receipt = json.loads(path.read_text())
                    receipt.update(
                        status="submitted"
                        if entry["id"] is not None
                        else entry["status"],
                        submitted=entry["id"] is not None,
                        uploaded=True,
                        submission_id=entry["id"],
                        submission_status=entry["status"],
                        public_score=entry.get("score"),
                        retry_at=entry.get("retry_at"),
                    )
                    write_json(path, receipt)
            if row.get("decision"):
                path = Path(row["decision"])
                decision = json.loads(path.read_text())
                decision["public_submission"] = snapshot
                write_json(path, decision)
            self.mark_round(row["id"], git_synced=False, git_commit=None)

    def defer_submission(self, identity, error):
        retry_at = (
            (datetime.now(timezone.utc) + timedelta(days=1))
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .strftime("%Y-%m-%dT%H:%M:%SZ")
        )

        def update(state):
            entry = state["submissions"][identity]
            if entry["id"] is not None:
                raise ValueError("Cannot defer an accepted submission")
            entry.setdefault("rejections", []).append({"at": now(), "reason": error})
            entry.update(status="quota_wait", error=error, retry_at=retry_at)
            if state["pending_submission"] == identity:
                state["pending_submission"] = None
            state["submit_after"] = retry_at

        self.change(update)
        self.publication_snapshot(identity)

    def update_submission(self, identity, status, score=None, error=None):
        previous = self.read()["submissions"][identity]
        changed = any(
            previous.get(key) != value
            for key, value in [("status", status), ("score", score), ("error", error)]
        )

        def update(s):
            entry = s["submissions"][identity]
            entry.update(status=status, score=score, error=error, checked_at=now())
            if status in ["COMPLETE", "ERROR", "CANCELLED"]:
                if s["pending_submission"] == identity:
                    s["pending_submission"] = None
                if score is not None and score > s["public_best"]["score"]:
                    s["public_best"] = {"score": score, "submission_id": entry["id"]}

        self.change(update)
        if changed:
            self.publication_snapshot(identity)

    def refresh_kaggle(self):
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()
        rows = api.competition_submissions(self.config["competition"], page_size=100)
        state = self.read()
        for identity, entry in state["submissions"].items():
            matching = next(
                (r for r in rows if entry["id"] is not None and r.ref == entry["id"]),
                None,
            )
            if matching is None and entry["status"] == "intent":
                # Reconcile a crash after remote acceptance before recording the returned ID.
                matching = next(
                    (
                        r
                        for r in rows
                        if r.description == entry["message"]
                        and r.submitted_by == "giaok246"
                    ),
                    None,
                )
                if matching:
                    self.finish_submission(identity, matching.ref)
                # An ambiguous intent remains reserved; never automatically submit again.
            if matching:
                status = str(matching.status).split(".")[-1]
                self.update_submission(
                    identity,
                    status,
                    float(matching.public_score) if matching.public_score else None,
                    matching.error_description,
                )
        import pandas as pd

        fields = [
            "ref",
            "fileName",
            "date",
            "description",
            "status",
            "publicScore",
            "privateScore",
        ]
        pd.DataFrame(
            [
                dict(
                    zip(
                        fields,
                        [
                            row.ref,
                            row.file_name,
                            str(row.date),
                            row.description,
                            str(row.status),
                            row.public_score,
                            row.private_score,
                        ],
                    )
                )
                for row in rows
            ]
        ).to_csv("results/kaggle_submissions.csv", index=False)
        return rows

    def import_bootstrap(self):
        identifier = self.config["bootstrap_submission"]
        identity = f"bootstrap-{identifier}"

        def update(s):
            if identity not in s["submissions"]:
                s["submissions"][identity] = {
                    "identity": identity,
                    "id": identifier,
                    "status": "PENDING",
                    "version": 4,
                    "message": "existing accepted chemistry submission",
                    "created_at": now(),
                }
                s["pending_submission"] = identity

        self.change(update)
        return self.refresh_kaggle()

    def mark_round(self, identifier, **values):
        self.change(
            lambda s: next(r for r in s["rounds"] if r["id"] == identifier).update(
                **values
            )
        )

    def evaluate_round(self, identifier):
        """Paired CSV evidence is mandatory; compare to the incumbent on identical keys."""
        import pandas as pd

        r = next(r for r in self.read()["rounds"] if r["id"] == identifier)
        if r.get("decision") and Path(r["decision"]).exists():
            return json.loads(Path(r["decision"]).read_text())
        directory = Path(r["report"]).parent
        report = json.loads(Path(r["report"]).read_text())
        if report.get("diagnostic_only"):
            result = {
                "round": identifier,
                "direction": r["direction"],
                "status": "diagnostic_complete",
                "report": report,
                "protocol_sha256": digest(directory / "protocol.json"),
                "winner": None,
                "independent_acceptance": False,
                "evaluation": "Development pilot diagnostics; no submission eligibility",
            }
            public = Path("results/research_loop") / f"{identifier}.json"
            write_json(public, result)
            self.mark_round(
                identifier, status="diagnostic_complete", decision=str(public)
            )
            return result
        cohort_path = Path(self.config["source"])
        usage = None
        if (cohort_path / "cohorts.json").exists():
            from casmi_ml.research_cohorts import record_usage

            usage = record_usage(
                self.root / "cohort_registry.json",
                cohort_path,
                "researchdev",
                identifier,
            )
        variants = list(report["unknown"])
        baseline_name = "legacy" if "legacy" in variants else "baseline"
        best = self.read()["development_best"]
        baseline_dir = Path(best["directory"]) if best else directory
        baseline_name = best["variant"] if best else baseline_name
        baseline = {}
        baseline_rows = {}
        for mode in ["unknown", "known"]:
            baseline_rows[mode] = pd.read_csv(
                baseline_dir / f"{mode}_{baseline_name}.csv"
            )
            baseline[mode] = (
                best["metrics"][mode] if best else report[mode][baseline_name]
            )
        choices = []
        for variant in variants:
            selected = {m: report[m][variant] for m in ["unknown", "known"]}
            folds = {}
            for mode in ["unknown", "known"]:
                frame = pd.read_csv(directory / f"{mode}_{variant}.csv")
                folds[mode] = fold_summary(
                    frame, baseline_rows[mode], self.config["development"]["folds"]
                )
                # Never accept an aggregate report that disagrees with its paired rows.
                if (
                    len(frame) != selected[mode]["molecules"]
                    or abs(frame.reciprocal_rank.mean() - selected[mode]["mrr25"])
                    > 1e-10
                    or abs(frame.top1.mean() - selected[mode]["top1"]) > 1e-10
                ):
                    raise ValueError("Metric report differs from paired CSV")
            choices.append(
                {
                    "variant": variant,
                    "metrics": selected,
                    "gate": development_gate(selected, baseline, self.config),
                    "folds": folds,
                }
            )
        eligible = [c for c in choices if c["gate"]["eligible"]]
        winner = (
            max(eligible, key=lambda c: c["metrics"]["unknown"]["mrr25"])
            if eligible
            else None
        )
        result = {
            "round": identifier,
            "round_directory": str(directory),
            "incumbent_directory": r["argv"][r["argv"].index("--incumbent") + 1]
            if "--incumbent" in r["argv"]
            else None,
            "direction": r["direction"],
            "cohort_usage": usage,
            "baseline": baseline,
            "choices": choices,
            "winner": winner,
            "independent_acceptance": False,
            "evaluation": "Repeated fixed molecule development cohort; no new holdout opened",
        }
        write_json(directory / "decision.json", result)
        public = Path("results/research_loop") / f"{identifier}.json"
        write_json(public, result)
        if winner:
            self.change(
                lambda s: s.update(
                    development_best={
                        "round": identifier,
                        "variant": winner["variant"],
                        "directory": str(directory),
                        "metrics": winner["metrics"],
                    }
                )
            )
        elif best is None:
            self.change(
                lambda s: s.update(
                    development_best={
                        "round": identifier,
                        "variant": baseline_name,
                        "directory": str(directory),
                        "metrics": baseline,
                    }
                )
            )
        self.mark_round(
            identifier,
            status="eligible" if winner else "rejected",
            decision=str(public),
        )
        return result

    def sync_github(self, identifier, paths):
        with (self.root / "github_sync.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._sync_github(identifier, paths)
            except subprocess.CalledProcessError as error:
                output = str(error.output or "")
                transient = [
                    "Could not resolve host",
                    "Failed to connect",
                    "Could not connect to server",
                    "Connection timed out",
                    "Connection reset",
                    "SSL connection timeout",
                    "requested URL returned error: 502",
                    "requested URL returned error: 503",
                    "requested URL returned error: 504",
                ]
                if not any(message.lower() in output.lower() for message in transient):
                    raise
                self.mark_round(
                    identifier,
                    github_sync_error=output[-2000:],
                    github_sync_checked_at=now(),
                )
                return False

    def _sync_github(self, identifier, paths):
        """Explicit curated paths only; push retries cannot duplicate a completed commit."""
        r = next(r for r in self.read()["rounds"] if r["id"] == identifier)
        if r["git_synced"]:
            return
        remote, branch = self.config["github_remote"], self.config["github_branch"]

        def git(*args):
            return subprocess.check_output(
                ["git", *args], text=True, stderr=subprocess.STDOUT
            ).strip()

        if git("branch", "--show-current") != branch:
            raise ValueError("Research sync must run on configured branch")
        git("fetch", remote, branch)
        if subprocess.run(
            ["git", "merge-base", "--is-ancestor", f"{remote}/{branch}", "HEAD"],
            check=False,
        ).returncode:
            raise RuntimeError("Remote advanced; reconcile without force before sync")
        if not r.get("git_commit"):
            if git("diff", "--cached", "--name-only"):
                raise RuntimeError(
                    "Existing staged edits must be handled before research sync"
                )
            git("add", "--", *paths)
            if git("diff", "--cached", "--name-only"):
                git(
                    "commit",
                    "-m",
                    f"Research {identifier}: verified experiment and aggregate results",
                )
            self.mark_round(identifier, git_commit=git("rev-parse", "HEAD"))
        git("push", remote, f"HEAD:{branch}")
        commit = git("rev-parse", "HEAD")
        self.mark_round(identifier, git_synced=True, github_sync_error=None)
        self.change(
            lambda s: s["github_commits"].append(
                {"round": identifier, "commit": commit}
            )
        )

    def publish_prepared(self, identifier, release):
        """Submit a checksum-verified, locally checked development experiment once."""
        from kaggle.api.kaggle_api_extended import KaggleApi

        release = Path(release)
        verification = json.loads((release / "verification.json").read_text())
        r = next(r for r in self.read()["rounds"] if r["id"] == identifier)
        decision = json.loads(Path(r["decision"]).read_text())
        if (
            not decision["winner"]
            or not decision["winner"]["gate"]["eligible"]
            or not verification.get("valid")
        ):
            raise ValueError(
                "Development gate and full inference verification required"
            )
        if (
            verification["seconds"] > self.config["inference_seconds"]
            or verification["peak_rss_mib"] > self.config["inference_rss_mib"]
        ):
            raise ValueError("Inference resource gate failed")
        sums = json.loads((release / "bundle/SHA256SUMS.json").read_text())
        if any(digest(release / "bundle" / name) != sha for name, sha in sums.items()):
            raise ValueError("Release content changed after verification")
        identity = release_identity(release, sums, decision["winner"]["variant"])
        state = self.read()
        if (
            state.get("submission_aliases", {}).get(identity, identity)
            != verification["identity"]
        ):
            raise ValueError("Verification belongs to different release contents")
        previous_identity = state.get("submission_aliases", {}).get(identity, identity)
        # Metadata-only amendments use the original verified canonical identity
        # for reservation as well as lookup, so their first submission is allowed
        # once while duplicates of already accepted contents remain impossible.
        identity = previous_identity
        previous = state["submissions"].get(previous_identity)
        if previous is not None and previous["status"] != "quota_wait":
            return previous
        metadata = json.loads((release / "notebook/kernel-metadata.json").read_text())
        api = KaggleApi()
        api.authenticate()
        remote = r.get("remote_release")
        if remote is None:
            # A round-specific private dataset avoids overwriting past deployment assets.
            dataset_id = json.loads(
                (release / "dataset/dataset-metadata.json").read_text()
            )["id"]
            if not r.get("dataset_uploaded"):
                response = api.dataset_create_new(
                    str(release / "dataset"),
                    public=False,
                    convert_to_csv=False,
                    quiet=True,
                )
                if response.error or str(response.status).lower() == "error":
                    raise ValueError(f"Kaggle rejected dataset: {response}")
                self.mark_round(
                    identifier, dataset_uploaded=True, dataset_response=str(response)
                )
            try:
                ready = api.dataset_status(dataset_id) == "ready"
            except Exception as error:
                from requests import ConnectionError, HTTPError, Timeout

                if isinstance(error, (ConnectionError, Timeout)):
                    self.mark_round(
                        identifier,
                        remote_status_error=str(error),
                        remote_status_checked_at=now(),
                    )
                    return {"status": "notebook_running"}
                if isinstance(error, HTTPError) and error.response.status_code in [
                    403,
                    404,
                ]:
                    return {"status": "notebook_running"}
                raise
            if not ready:
                return {"status": "notebook_running"}
            response = api.kernels_push(str(release / "notebook"), timeout="1800")
            if (
                response.error
                or response.invalid_dataset_sources
                or response.invalid_competition_sources
            ):
                raise ValueError(f"Kaggle rejected notebook inputs: {response}")
            remote = {
                "kernel": kernel_ref(response.ref or metadata["id"]),
                "version": int(response.version_number),
            }
            self.mark_round(identifier, remote_release=remote)
        try:
            status = api.kernels_status(remote["kernel"])
        except Exception as error:
            from requests import ConnectionError, Timeout

            if not isinstance(error, (ConnectionError, Timeout)):
                raise
            # This call is read-only. Keep the remote version and continue local
            # research; never retry an uncertain competition POST here.
            self.mark_round(
                identifier,
                remote_status_error=str(error),
                remote_status_checked_at=now(),
            )
            return {"status": "notebook_running"}
        status = (
            str(status["status"] if isinstance(status, dict) else status.status)
            .split(".")[-1]
            .upper()
        )
        if status == "ERROR":
            self.mark_round(identifier, status="release_failed")
            raise RuntimeError("Kaggle notebook failed; inspect before retrying")
        if status != "COMPLETE":
            return {"status": "notebook_running"}
        state = self.read()
        if state.get("submit_after") and now() < state["submit_after"]:
            return {"status": "waiting_for_allowance"}
        if state["pending_submission"] is not None:
            return {"status": "waiting_for_previous_submission"}
        message = f"Research {identifier} dev-experimental {identity[:16]}"
        self.reserve_submission(identity, remote["version"], message)
        try:
            response = api.competition_submit_code(
                "submission.csv",
                message,
                competition=self.config["competition"],
                kernel=remote["kernel"],
                kernel_version=remote["version"],
            )
        except Exception as error:
            from requests import HTTPError

            if (
                isinstance(error, HTTPError)
                and error.response.status_code == 400
                and "daily Submission allowance" in error.response.text
            ):
                self.mark_round(identifier, identity=identity)
                self.defer_submission(
                    identity,
                    "Kaggle daily submission allowance exhausted; rejected before acceptance",
                )
                return {"status": "waiting_for_allowance"}
            # Uncertain errors retain intent for reconciliation, never resubmit.
            raise
        self.finish_submission(identity, response.ref)
        self.mark_round(identifier, identity=identity, status="submitted")
        self.publication_snapshot(identity)
        return self.read()["submissions"][identity]

    def status(self):
        return self.read()

    def public_paths(self):
        plan = json.loads(Path("configs/research_publish_paths.json").read_text())
        paths = [p for p in plan["paths"] if Path(p).is_file()]
        for directory in plan["aggregate_directories"]:
            paths.extend(str(p) for p in Path(directory).rglob("*.json"))
        return paths

    def complete_round(self, identifier):
        r = next(r for r in self.read()["rounds"] if r["id"] == identifier)
        if r["status"] == "eligible" and self.config["automatic_submission"]:
            if r["direction"] in [
                "mass",
                "generation_slots",
                "coverage",
                "reference_guard",
                "reference_generation",
                "generation_position",
                "generation_model",
                "generated_frequency",
                "generated_position_update",
                "generated_second_reference",
                "generated_expanded_route",
                "generated_first_gate",
                "protected_generation",
            ]:
                outcome = self.release_round(identifier)
                if outcome in [
                    "notebook_running",
                    "waiting_for_previous_submission",
                    "waiting_for_allowance",
                ]:
                    updated = next(
                        row for row in self.read()["rounds"] if row["id"] == identifier
                    )
                    if (
                        self.config["automatic_github_sync"]
                        and not updated["git_synced"]
                    ):
                        self.sync_github(identifier, self.public_paths())
                    return "publication_waiting"
                if outcome in ["stopped", "budget_exhausted"]:
                    return outcome
            else:
                # Each extension needs its own matching inference path before remote publication.
                self.mark_round(identifier, status="deployment_needed")
                if self.config["automatic_github_sync"]:
                    self.sync_github(identifier, self.public_paths())
                return "deployment_needed"
        r = next(row for row in self.read()["rounds"] if row["id"] == identifier)
        if (
            self.config["automatic_github_sync"]
            and not r["git_synced"]
            and self.sync_github(identifier, self.public_paths()) is False
        ):
            return "github_waiting"
        return "round_complete"

    def release_round(self, identifier):
        from casmi_ml.experimental_release import package

        r = next(r for r in self.read()["rounds"] if r["id"] == identifier)
        release = self.root / "releases" / identifier
        if r.get("release"):
            release = Path(r["release"])
        decision = json.loads(Path(r["decision"]).read_text())
        if r["direction"] in [
            "reference_guard",
            "reference_generation",
            "generation_position",
            "generation_model",
            "generated_frequency",
            "generated_position_update",
            "generated_second_reference",
            "generated_expanded_route",
            "generated_first_gate",
        ]:
            directory = Path(r["report"]).parent
            replay = directory / "replay.json"
            if not replay.exists():
                gpu = (self.root / "gpu.lock").open("a")
                try:
                    try:
                        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return "notebook_running"
                    result = self.job(
                        [
                            ".venv-gpu/bin/python",
                            "-m",
                            "casmi_ml.reference_replay",
                            "--directory",
                            str(directory),
                        ],
                        directory / "replay.log",
                        1800,
                        auxiliary_key="replay-" + identifier,
                    )
                    if result != "complete":
                        return result
                finally:
                    gpu.close()
        if r["direction"] == "coverage":
            replay = Path(r["report"]).parent / "replay.json"
            if not replay.exists():
                result = self.job(
                    [
                        ".venv/bin/python",
                        "-m",
                        "casmi_ml.coverage_replay",
                        "--directory",
                        str(Path(r["report"]).parent),
                    ],
                    replay.with_suffix(".log"),
                    1800,
                    auxiliary_key="replay-" + identifier,
                )
                if result != "complete":
                    return result
        if r["direction"] in ["generation_slots", "protected_generation"]:
            directory = Path(r["report"]).parent
            replay = directory / "replay/verification.json"
            protocol = json.loads((directory / "protocol.json").read_text())
            _, prefix, slots = decision["winner"]["variant"].split("_")
            if not replay.exists():
                from casmi_ml.research_protocol import ROOT

                checkpoint = Path(
                    protocol.get(
                        "generator_checkpoint", ROOT / "generation/smiles_42/model.pt"
                    )
                )
                suffix = (
                    ""
                    if checkpoint.resolve()
                    == (ROOT / "generation/smiles_42/model.pt").resolve()
                    else "_" + digest(checkpoint)[:12]
                )
                samples = (
                    ROOT
                    / "generation"
                    / f"researchdev_samples128_limitall_stable_v2{suffix}.json"
                )
                gpu = (self.root / "gpu.lock").open("a")
                try:
                    try:
                        fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        return "notebook_running"
                    result = self.job(
                        [
                            ".venv-gpu/bin/python",
                            "-m",
                            "casmi_ml.generation_replay",
                            "--generated",
                            str(samples),
                            "--incumbent",
                            str(
                                protocol.get(
                                    "mass_directory",
                                    "artifacts/research_loop/rounds/0001_mass_v2",
                                )
                            )
                            if r["direction"] == "protected_generation"
                            else decision["incumbent_directory"],
                            "--output",
                            str(replay.parent),
                            "--prefix",
                            prefix,
                            "--slots",
                            slots,
                            "--checkpoint",
                            str(checkpoint),
                            *(
                                ["--open-protected"]
                                if r["direction"] == "protected_generation"
                                else []
                            ),
                        ],
                        directory / "replay.log",
                        1800,
                        auxiliary_key="replay-" + identifier,
                    )
                    if result != "complete":
                        return result
                finally:
                    gpu.close()
        if not release.exists():
            package(identifier, r["decision"], release)
        if not (release / "verification.json").exists():
            gpu = None
            if r["direction"] in [
                "generation_slots",
                "reference_guard",
                "reference_generation",
                "generation_position",
                "generation_model",
                "generated_frequency",
                "generated_position_update",
                "generated_second_reference",
                "generated_expanded_route",
                "generated_first_gate",
                "protected_generation",
            ]:
                gpu = (self.root / "gpu.lock").open("a")
                try:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    gpu.close()
                    return "notebook_running"
            try:
                result = self.job(
                    [
                        ".venv-gpu/bin/python"
                        if r["direction"]
                        in [
                            "generation_slots",
                            "reference_guard",
                            "reference_generation",
                            "generation_position",
                            "generation_model",
                            "generated_frequency",
                            "generated_position_update",
                            "generated_second_reference",
                            "generated_expanded_route",
                            "generated_first_gate",
                            "protected_generation",
                        ]
                        else ".venv/bin/python",
                        "-m",
                        "casmi_ml.experimental_release",
                        "--verify",
                        "--output",
                        str(release),
                    ],
                    release / "verification.log",
                    self.config["inference_seconds"],
                    auxiliary_key="verification-" + identifier,
                )
            finally:
                if gpu is not None:
                    gpu.close()
            if result != "complete":
                return result
        self.mark_round(identifier, release=str(release))
        result = self.publish_prepared(identifier, release)
        return result["status"]

    def step(self):
        if self.stopped():
            self.change(lambda s: s.update(status="stopped"))
            return "stopped"
        state = self.read()
        if state["pending_submission"] is not None:
            try:
                self.refresh_kaggle()
            except Exception as error:
                from requests import ConnectionError, Timeout

                if not isinstance(error, (ConnectionError, Timeout)):
                    raise
                message = str(error)
                self.change(
                    lambda s: s.update(
                        remote_refresh_error=message, remote_refresh_checked_at=now()
                    )
                )
            state = self.read()
        finished = next(
            (
                r
                for r in state["rounds"]
                if r["status"]
                in [
                    "eligible",
                    "rejected",
                    "submitted",
                    "failed",
                    "release_failed",
                    "diagnostic_complete",
                    "recorded",
                ]
                and (not r["git_synced"] or r["status"] == "eligible")
            ),
            None,
        )
        waiting = False
        if finished:
            outcome = self.complete_round(finished["id"])
            if outcome != "publication_waiting":
                return outcome
            waiting = True
            # Preparing later eligible releases can overlap a platform scoring
            # wait. publish_prepared still enforces one pending submission.
            for later in state["rounds"]:
                if later["id"] == finished["id"]:
                    continue
                if later["status"] != "eligible" and (
                    later["git_synced"]
                    or later["status"]
                    not in [
                        "rejected",
                        "submitted",
                        "failed",
                        "release_failed",
                        "diagnostic_complete",
                        "recorded",
                    ]
                ):
                    continue
                outcome = self.complete_round(later["id"])
                if outcome != "publication_waiting":
                    return outcome
        evaluated = next(
            (r for r in state["rounds"] if r["status"] == "evaluated"), None
        )
        if evaluated:
            self.evaluate_round(evaluated["id"])
            return "decided"
        pending = next(
            (
                r
                for r in state["rounds"]
                if r["status"] in ["queued", "running", "retry"]
            ),
            None,
        )
        if pending is None:
            if waiting or self.read()["pending_submission"] is not None:
                return "publication_waiting"
            for entry in self.read().get("auxiliary_jobs", {}).values():
                proc = Path(f"/proc/{entry['pid']}/cmdline")
                try:
                    actual = [v.decode() for v in proc.read_bytes().split(b"\0") if v]
                except FileNotFoundError:
                    continue
                if actual == entry["argv"]:
                    return "auxiliary_running"
            self.change(lambda s: s.update(status="research_needed"))
            return "research_needed"
        rid = pending["id"]

        def mark(**values):
            self.change(
                lambda s: next(r for r in s["rounds"] if r["id"] == rid).update(
                    **values
                )
            )

        if Path(pending["report"]).exists():
            mark(status="evaluated", completed_at=now())
            return "evaluated"
        if pending["direction"] in [
            "generation_slots",
            "generation_pilot",
            "representation",
            "generation_finetune",
            "generation_model",
            "generated_frequency",
            "generated_position_update",
            "generated_second_reference",
            "generated_expanded_route",
            "generated_first_gate",
        ]:
            # The child owns the GPU budget/lock. Probe availability before
            # launching it so a queued GPU job remains queued during another
            # stage instead of failing merely because the GPU is busy.
            with (self.root / "gpu.lock").open("a") as gpu:
                try:
                    fcntl.flock(gpu, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    return "auxiliary_running"
        mark(status="running")
        active = state.get("active_job")
        if active:
            # A controller can die while its child survives. Never launch a duplicate.
            try:
                os.kill(active["pid"], 0)
            except ProcessLookupError:
                self.change(lambda s: s.update(active_job=None))
                if Path(pending["report"]).exists():
                    mark(status="evaluated", completed_at=now())
                    return "evaluated"
            else:
                return "orphan_job_running"
        seconds = self.config["gpu_stage_seconds"].get(pending["direction"])
        if pending["direction"] in [
            "generation_slots",
            "generation_pilot",
            "generation_finetune",
            "generation_model",
            "generated_frequency",
            "generated_position_update",
            "generated_second_reference",
            "generated_expanded_route",
            "generated_first_gate",
        ]:
            seconds = self.config["gpu_stage_seconds"]["generation"]
        try:
            result = self.job(
                pending["argv"], self.root / "rounds" / rid / "execution.log", seconds
            )
        except Exception as error:
            mark(status="failed", error=str(error), completed_at=now())
            write_json(
                Path("results/research_loop") / f"{rid}.json",
                {"round": rid, "status": "failed", "error": str(error)},
            )
            raise
        if result == "budget_exhausted":
            mark(
                status="failed", error="wall_time_budget_exhausted", completed_at=now()
            )
            return "budget_exhausted"
        if result == "stopped":
            mark(status="retry")
            return "stopped"
        if not Path(pending["report"]).exists():
            raise ValueError("Experiment completed without report")
        mark(status="evaluated", completed_at=now())
        return "evaluated"

    def run(self, once=False):
        with (self.root / "runner.lock").open("a") as lock:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise RuntimeError("A research runner is active") from error
            while True:
                result = self.step()
                if result in [
                    "publication_waiting",
                    "github_waiting",
                    "orphan_job_running",
                    "auxiliary_running",
                ]:
                    time.sleep(min(45, self.config["poll_seconds"]))
                if once or result in [
                    "stopped",
                    "research_needed",
                    "deployment_needed",
                ]:
                    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "command", choices=["run", "status", "stop", "resume", "run-auxiliary"]
    )
    p.add_argument("--config", type=Path, default=CONFIG)
    p.add_argument("--once", action="store_true")
    p.add_argument("--round")
    a = p.parse_args()
    c = Controller(a.config)
    if a.command == "run-auxiliary":
        if not a.round:
            p.error("--round required")
        print(c.run_auxiliary(a.round))
    elif a.command == "stop":
        c.stop()
        print(
            "Stop requested; child process group will be terminated and state retained"
        )
    elif a.command == "resume":
        c.resume()
        print(c.run(a.once))
    elif a.command == "run":
        print(c.run(a.once))
    else:
        print(json.dumps(c.status(), indent=2))


if __name__ == "__main__":
    main()
