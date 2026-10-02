import copy
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml.chemistry import parse_adduct
from casmi_ml.mass_candidates import candidate_window, mass_centers
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_loop import (
    Controller,
    content_identity,
    development_gate,
    fold_summary,
)

CONFIG = json.loads(Path("configs/research_loop.json").read_text())


def sample(n=2000, mrr=0.025, known=0.55, top1=0.5):
    return {
        "unknown": {"molecules": n, "mrr25": mrr, "top1": 0.01},
        "known": {"molecules": 1400, "mrr25": known, "top1": top1},
    }


class LoopTests(unittest.TestCase):
    def controller(self, root):
        config = copy.deepcopy(CONFIG)
        config["root"] = str(root / "state")
        path = root / "config.json"
        path.write_text(json.dumps(config))
        return Controller(path)

    def test_gate_boundaries_and_rejects_nonfinite_protection(self):
        base = sample()
        self.assertTrue(
            development_gate(sample(mrr=0.0255, known=0.549, top1=0.495), base, CONFIG)[
                "eligible"
            ]
        )
        for bad in [
            sample(1999, 0.03),
            sample(mrr=0.02549),
            sample(mrr=0.03, known=0.5489),
            sample(mrr=0.03, top1=0.4949),
            sample(mrr=float("nan")),
        ]:
            self.assertFalse(development_gate(bad, base, CONFIG)["eligible"])
        base["known"]["mrr25"] = float("nan")
        self.assertFalse(development_gate(sample(mrr=0.03), base, CONFIG)["eligible"])

    def test_requires_same_cohort_size(self):
        self.assertIn(
            "cohort_size_mismatch",
            development_gate(sample(n=2100, mrr=0.03), sample(), CONFIG)["reasons"],
        )

    def test_submission_intent_survives_restart_and_public_best_survives_regression(
        self,
    ):
        with tempfile.TemporaryDirectory() as d:
            c = self.controller(Path(d))
            c.reserve_submission("sha", 1, "unique message")
            c = Controller(c.config_path)
            with self.assertRaises(RuntimeError):
                c.reserve_submission("sha2", 2, "other")
            c.finish_submission("sha", 123)
            c.update_submission("sha", "COMPLETE", 0.173)
            self.assertEqual(c.read()["public_best"]["score"], 0.176)
            with self.assertRaises(RuntimeError):
                c.reserve_submission("sha", 1, "unique message")
            c.reserve_submission("sha2", 2, "other")
            c.finish_submission("sha2", 124)
            c.update_submission("sha2", "COMPLETE", 0.18)
            self.assertEqual(c.read()["public_best"]["submission_id"], 124)

    def test_stop_blocks_new_jobs_and_wall_deadline_terminates_child(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = self.controller(root)
            c.stop()
            self.assertEqual(c.job(["false"], root / "log"), "stopped")
            c.resume()
            self.assertEqual(
                c.job(["sleep", "10"], root / "log", seconds=0.03), "budget_exhausted"
            )
            self.assertIsNone(c.read()["active_job"])

    def test_paired_folds_validate_keys_and_reproduce(self):
        frame = pd.DataFrame(
            {"key": [f"key{i}" for i in range(100)], "reciprocal_rank": [0.1] * 100}
        )
        a = fold_summary(frame, frame)
        self.assertEqual(a, fold_summary(frame.sample(frac=1, random_state=42), frame))
        self.assertEqual(sum(r["molecules"] for r in a["folds"]), 100)
        with self.assertRaises(ValueError):
            fold_summary(frame.iloc[:-1], frame)
        with self.assertRaises(ValueError):
            fold_summary(pd.concat([frame, frame.iloc[:1]]), frame)

    def test_checksum_identity_detects_code_and_config_changes(self):
        with tempfile.TemporaryDirectory() as d:
            file = Path(d) / "model"
            file.write_bytes(b"one")
            a = content_identity([file], {"weight": 0.5})
            self.assertNotEqual(a, content_identity([file], {"weight": 0.6}))
            file.write_bytes(b"two")
            self.assertNotEqual(a, content_identity([file], {"weight": 0.5}))

    def test_crash_reservation_and_global_gpu_exclusion(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict("os.environ", {"CASMI_GPU_LOCK": str(Path(d) / "gpu.lock")}),
        ):
            root = Path(d)
            with patch("casmi_ml.research_budget.time.monotonic", return_value=0):
                a = StageBudget(root, "representation", "one", 8, limit=10)
            with self.assertRaises(BlockingIOError):
                StageBudget(root / "other", "generation", "one", 8, limit=10)
            with patch("casmi_ml.research_budget.time.monotonic", return_value=6):
                a.close()
            with patch("casmi_ml.research_budget.time.monotonic", return_value=6):
                b = StageBudget(root, "representation", "two", 8, limit=10)
            self.assertEqual(b.allowance, 4)
            # Simulate SIGKILL: close OS lock without clean budget refund.
            b.lock.close()
            with self.assertRaises(RuntimeError):
                StageBudget(root, "representation", "three", 8, limit=10)

    def test_budget_is_durable_after_real_sigkill(self):
        with (
            tempfile.TemporaryDirectory() as d,
            patch.dict("os.environ", {"CASMI_GPU_LOCK": str(Path(d) / "gpu.lock")}),
        ):
            code = (
                "from casmi_ml.research_budget import StageBudget; "
                f'b=StageBudget({d!r}, "generation", "killed", 10, limit=10); '
                "import os,signal; os.kill(os.getpid(),signal.SIGKILL)"
            )
            subprocess.run(
                [".venv/bin/python", "-c", code],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            with self.assertRaises(RuntimeError):
                StageBudget(d, "generation", "retry", 10, limit=10)


class MassTests(unittest.TestCase):
    def test_multicharge_and_dimer_mass(self):
        rows = []
        for adduct in ["[M+2H]2+", "[2M+H]+", "[M+Na]+"]:
            ion = parse_adduct(adduct)
            rows.append(
                {
                    "adduct": adduct,
                    "precursor_mz": (300 * ion.molecules + ion.shift) / abs(ion.charge),
                }
            )
        group = pd.DataFrame(rows)
        self.assertAlmostEqual(mass_centers(group, "charge_aware_median")[0], 300)
        for mass in mass_centers(group, "charge_aware_union"):
            self.assertAlmostEqual(mass, 300)

    def test_union_deduplicates_candidates(self):
        class Index:
            catalog = pd.DataFrame({"inchikey14": ["a", "b"]})

            def query(self, mass):
                return self.catalog

            def fps(self, frame):
                return frame, None

        group = pd.DataFrame(
            {"precursor_mz": [301.007276, 302.007276], "adduct": ["[M+H]+", "[M+H]+"]}
        )
        pool, _ = candidate_window(Index(), group, "charge_aware_union")
        self.assertEqual(pool.inchikey14.tolist(), ["a", "b"])


if __name__ == "__main__":
    unittest.main()


class ReferenceWindowTests(unittest.TestCase):
    def test_explicit_centers_preserve_original_query_copy_exclusion(self):
        from casmi_ml.ranking import build_reference

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            catalog = pd.DataFrame(
                {
                    "inchikey14": ["a"],
                    "normalized_smiles": ["CCO"],
                    "mass": [600.0],
                    "split": ["train"],
                }
            )
            train = pd.DataFrame(
                [
                    {
                        "inchikey14": "a",
                        "normalized_smiles": "CCO",
                        "ms2_mzs": [peak],
                        "ms2_normalized_intensities": [1.0],
                        "precursor_error_ppm": 0.0,
                    }
                    for peak in [42.0, 43.0]
                ]
            )
            train.to_parquet(root / "train.parquet")
            query = train.iloc[:1].copy()
            query["adduct"] = "[M+H]+"
            query["precursor_mz"] = 301.007276
            build_reference(
                root / "train.parquet",
                catalog,
                query,
                root / "ref",
                exclude_queries=True,
                target_centers=[600.0, float("nan")],
            )
            rows = pd.read_parquet(root / "ref/rows.parquet")
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows.mass.tolist(), [600.0])


class GeneratedSlotTests(unittest.TestCase):
    def test_only_novel_unique_candidates_insert_and_base_order_preserved(self):
        from casmi_ml.generation_slots import insert_generated

        base = [f"b{i}" for i in range(30)]
        result = insert_generated(base, ["b1", "new", "new", "next"], 5, 1)
        self.assertEqual(result[:6], base[:5] + ["new"])
        self.assertEqual([key for key in result if key != "new"], base)
        self.assertNotIn("next", result)
        self.assertEqual(insert_generated(base, [], 5, 3), base)


class ReleaseEvidenceTests(unittest.TestCase):
    def test_relocated_release_has_same_content_identity(self):
        import shutil

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "a/pkg").mkdir(parents=True)
            (root / "a/pkg/code.py").write_bytes(b"code")
            (root / "a/model.pt").write_bytes(b"model")
            shutil.copytree(root / "a", root / "b")
            self.assertEqual(
                content_identity([root / "a/pkg/code.py", root / "a/model.pt"], {}),
                content_identity([root / "b/pkg/code.py", root / "b/model.pt"], {}),
            )

    def test_kernel_response_path_normalizes_to_owner_slug(self):
        from casmi_ml.research_loop import kernel_ref

        self.assertEqual(kernel_ref("/code/owner/slug"), "owner/slug")
        self.assertEqual(
            kernel_ref("https://www.kaggle.com/code/owner/slug"), "owner/slug"
        )

    def test_existing_submission_identity_cannot_submit_again(self):
        from unittest.mock import MagicMock

        from casmi_ml.data import write_json

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = LoopTests().controller(root)
            release = root / "release"
            (release / "bundle").mkdir(parents=True)
            file = release / "bundle/code.py"
            file.write_bytes(b"code")
            from casmi_ml.metfrag import digest

            write_json(release / "bundle/SHA256SUMS.json", {"code.py": digest(file)})
            identity = content_identity([file], {"variant": "charge_aware_union"})
            write_json(
                release / "verification.json",
                {"valid": True, "identity": identity, "seconds": 1, "peak_rss_mib": 1},
            )
            decision = root / "decision.json"
            write_json(
                decision,
                {
                    "winner": {
                        "variant": "charge_aware_union",
                        "gate": {"eligible": True},
                    }
                },
            )
            c.register_round("round", "mass", [], root / "report.json")
            c.mark_round("round", decision=str(decision))
            c.reserve_submission(identity, 1, "once")
            c.finish_submission(identity, 123)
            c.update_submission(identity, "COMPLETE", 0.173)
            with patch(
                "kaggle.api.kaggle_api_extended.KaggleApi", new=MagicMock()
            ) as api:
                result = c.publish_prepared("round", release)
            self.assertEqual(result["id"], 123)
            api.assert_not_called()


class CohortAndSamplingTests(unittest.TestCase):
    def test_sampling_ignores_label_and_processing_order(self):
        from casmi_ml.generation_sampling import sampling_seed

        frame = pd.DataFrame(
            {
                "adduct": ["[M+H]+", "[M+Na]+"],
                "precursor_mz": [101.0, 123.0],
                "ms2_mzs": [[42.0], [43.0]],
                "ms2_normalized_intensities": [[1.0], [1.0]],
                "inchikey14": ["secret", "secret"],
                "molecule_id": ["id", "id"],
            }
        )
        seed = sampling_seed(frame)
        frame["inchikey14"] = "other_truth"
        frame["molecule_id"] = "different_id"
        self.assertEqual(seed, sampling_seed(frame.iloc[::-1]))
        frame.loc[0, "precursor_mz"] += 0.01
        self.assertNotEqual(seed, sampling_seed(frame))

    def test_cohort_registry_marks_repeated_use(self):
        from casmi_ml.data import write_json
        from casmi_ml.metfrag import digest
        from casmi_ml.research_cohorts import record_usage

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            (root / "researchdev.parquet").write_bytes(b"frozen parquet stand-in")
            write_json(
                root / "cohorts.json",
                {
                    "researchdev": {
                        "keys": ["a", "b"],
                        "molecules": 2,
                        "sha256": digest(root / "researchdev.parquet"),
                    }
                },
            )
            one = record_usage(root / "registry.json", root, "researchdev", "first")
            two = record_usage(root / "registry.json", root, "researchdev", "second")
            self.assertEqual(one["fresh_molecules"], 2)
            self.assertEqual(two["previously_used_molecules"], 2)
            self.assertEqual(
                two, record_usage(root / "registry.json", root, "researchdev", "second")
            )


class AuxiliaryJobTests(unittest.TestCase):
    def test_auxiliary_job_preserves_existing_cpu_job_and_stop_terminates_it(self):
        import threading
        import time

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = LoopTests().controller(root)
            primary = {"pid": 987654321, "argv": ["primary"], "started_at": "test"}
            c.change(lambda s: s.update(active_job=primary))
            result = []
            worker = threading.Thread(
                target=lambda: result.append(
                    c.job(["sleep", "20"], root / "log", 20, auxiliary_key="gpu")
                )
            )
            worker.start()
            deadline = time.monotonic() + 3
            while not c.read().get("auxiliary_jobs") and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertEqual(c.read()["active_job"], primary)
            c.stop()
            worker.join(timeout=3)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, ["stopped"])
            self.assertFalse(c.read()["auxiliary_jobs"])
            self.assertEqual(c.read()["active_job"], primary)


class CoverageFusionTests(unittest.TestCase):
    def test_fragment_budget_exhaustion_preserves_original_ranking(self):
        from casmi_ml.coverage_inference import merge_expanded

        self.assertEqual(
            merge_expanded(["a", "b"], ["b", "c"], 1.0, fallback=True), ["a", "b"]
        )
        self.assertEqual(merge_expanded(["a", "b"], ["b", "c"], 1.0)[0], "b")
        with self.assertRaises(ValueError):
            merge_expanded(["a"], ["b"], 1.1)


class AtomicReleaseTests(unittest.TestCase):
    def test_failed_packaging_does_not_leave_partial_release(self):
        from casmi_ml.experimental_release import package

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)

            def fail(identifier, decision, staged):
                staged.mkdir()
                (staged / "model.pt").write_bytes(b"incomplete")
                raise ValueError("Replay failed")

            with (
                patch("casmi_ml.experimental_release._package", side_effect=fail),
                self.assertRaises(ValueError),
            ):
                package("round", root / "decision.json", root / "release")
            self.assertFalse((root / "release").exists())
            self.assertEqual(list(root.iterdir()), [])


class ReferenceEvidenceTests(unittest.TestCase):
    def test_guard_depends_on_reference_presence_without_truth(self):
        from casmi_ml.reference_guard import protects_reference

        self.assertTrue(protects_reference(["known", "external"], {"known"}, 0.0))
        self.assertFalse(protects_reference(["external", "known"], {"known"}, 0.0))
        self.assertTrue(protects_reference(["external"], set(), 0.9))
        self.assertFalse(protects_reference([], {"known"}, 0.0))


class JobBudgetScopeTests(unittest.TestCase):
    def test_replay_and_experiment_can_freeze_different_deadlines(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = LoopTests().controller(root)
            self.assertEqual(c.job(["true"], root / "execution.log", 10), "complete")
            self.assertEqual(c.job(["true"], root / "replay.log", 5), "complete")
            ledger = json.loads((root / "training_budget.json").read_text())
            self.assertEqual(ledger["wall_job:execution.log"]["limit_seconds"], 10)
            self.assertEqual(ledger["wall_job:replay.log"]["limit_seconds"], 5)


class PublicationPreparationTests(unittest.TestCase):
    def test_later_release_prepares_while_first_waits_for_score(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = LoopTests().controller(root)
            for rid in ["first", "second"]:
                c.register_round(rid, "mass", [], root / rid / "report.json")
                c.mark_round(rid, status="eligible", git_synced=True)
            with patch.object(
                c, "complete_round", return_value="publication_waiting"
            ) as release:
                self.assertEqual(c.step(), "publication_waiting")
            self.assertEqual(
                [call.args[0] for call in release.call_args_list], ["first", "second"]
            )


class SubmissionSnapshotTests(unittest.TestCase):
    def test_new_score_refreshes_public_receipt_and_requests_github_sync(self):
        from casmi_ml.data import write_json

        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            c = LoopTests().controller(root)
            release = root / "release"
            write_json(release / "status.json", {"independent_acceptance": False})
            decision = root / "decision.json"
            write_json(decision, {"winner": None})
            c.register_round("round", "mass", [], root / "report.json")
            c.mark_round(
                "round",
                identity="sha",
                release=str(release),
                decision=str(decision),
                git_synced=True,
                git_commit="old",
            )
            c.reserve_submission("sha", 1, "message")
            c.finish_submission("sha", 123)
            c.update_submission("sha", "COMPLETE", 0.173)
            self.assertFalse(c.read()["rounds"][0]["git_synced"])
            self.assertIsNone(c.read()["rounds"][0]["git_commit"])
            self.assertEqual(
                json.loads((release / "status.json").read_text())["public_score"], 0.173
            )
            self.assertEqual(
                json.loads(decision.read_text())["public_submission"]["id"], 123
            )
            c.mark_round("round", git_synced=True)
            c.update_submission("sha", "COMPLETE", 0.173)
            self.assertTrue(c.read()["rounds"][0]["git_synced"])
