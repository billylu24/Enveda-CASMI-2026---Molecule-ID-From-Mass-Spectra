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
            write_json(decision, {"winner": {"variant": "charge_aware_union"}})
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
