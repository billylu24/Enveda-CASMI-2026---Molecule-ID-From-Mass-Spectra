import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd

from casmi_ml.data import write_json
from casmi_ml.one_time_kaggle import amend_unsubmitted, authorize, reserve, step
from casmi_ml.research_loop import Controller


class OneTimePublicationTests(unittest.TestCase):
    def fixture(self, root):
        config = json.loads(Path("configs/research_loop.json").read_text())
        config["root"] = str(root / "state")
        write_json(root / "config.json", config)
        c = Controller(root / "config.json")
        c.change(lambda state: state.update(local_only=True))
        release = root / "release"
        return c, release

    def test_authorization_keeps_local_only_and_cannot_change_release(self):
        with tempfile.TemporaryDirectory() as directory:
            c, release = self.fixture(Path(directory))
            with patch("casmi_ml.one_time_kaggle.check_release", return_value="one"):
                authorize(release, c)
                authorize(release, c)
            self.assertTrue(c.local_only())
            self.assertEqual(
                c.read()["one_time_kaggle"]["maximum_accepted_submissions"], 1
            )
            with (
                patch("casmi_ml.one_time_kaggle.check_release", return_value="two"),
                self.assertRaisesRegex(ValueError, "another release"),
            ):
                authorize(release, c)

    def amendment_fixture(self, root):
        c, old = self.fixture(root)
        new = root / "repaired"
        for release in (old, new):
            (release / "bundle").mkdir(parents=True)
            (release / "local_output").mkdir()
            write_json(release / "bundle/SHA256SUMS.json", {"model.pt": "same"})
            write_json(release / "bundle/deployment_recipe.json", {"science": 236})
            pd.DataFrame({"molecule_id": range(400), "rank1": ["CC"] * 400}).to_csv(
                release / "local_output/submission.csv", index=False
            )
        with patch("casmi_ml.one_time_kaggle.check_release", return_value="one"):
            authorize(old, c)
        c.change(
            lambda state: state["one_time_kaggle"].update(
                status="platform_verification_failed"
            )
        )
        return c, old, new

    def test_runtime_amendment_preserves_authorization_and_failed_history(self):
        with tempfile.TemporaryDirectory() as directory:
            c, _old, new = self.amendment_fixture(Path(directory))
            previous = c.read()["one_time_kaggle"]
            with patch("casmi_ml.one_time_kaggle.check_release", return_value="two"):
                result = amend_unsubmitted(new, c)
            self.assertTrue(c.local_only())
            self.assertEqual(result["identity"], "two")
            self.assertEqual(result["status"], "authorized")
            self.assertEqual(result["authorized_at"], previous["authorized_at"])
            self.assertEqual(result["maximum_accepted_submissions"], 1)
            self.assertFalse(result["automatic_future_submissions"])
            self.assertFalse(result["github_pushes"])
            self.assertEqual(result["amendments"], [previous])
            self.assertIsNone(c.read()["pending_submission"])

    def test_amendment_refuses_accepted_or_uncertain_submission(self):
        for change in ("id", "pending", "intent", "status", "local_policy"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                c, _old, new = self.amendment_fixture(Path(directory))

                def mutate(state, change=change):
                    if change == "id":
                        state["one_time_kaggle"]["submission_id"] = 123
                    elif change == "pending":
                        state["pending_submission"] = "one"
                    elif change == "intent":
                        state["submissions"]["one"] = {"status": "intent"}
                    elif change == "status":
                        state["one_time_kaggle"]["status"] = "submission_intent"
                    else:
                        state["local_only"] = False

                c.change(mutate)
                before = c.read()
                with (
                    patch("casmi_ml.one_time_kaggle.check_release", return_value="two"),
                    self.assertRaises(ValueError),
                ):
                    amend_unsubmitted(new, c)
                self.assertEqual(c.read(), before)

    def test_amendment_refuses_changed_science_or_complete_rankings(self):
        for change in ("model", "added_model", "recipe", "rank"):
            with (
                self.subTest(change=change),
                tempfile.TemporaryDirectory() as directory,
            ):
                c, _old, new = self.amendment_fixture(Path(directory))
                if change in ("model", "added_model"):
                    sums = (
                        {"model.pt": "changed"}
                        if change == "model"
                        else {"model.pt": "same", "extra.pt": "new"}
                    )
                    write_json(new / "bundle/SHA256SUMS.json", sums)
                elif change == "recipe":
                    write_json(new / "bundle/deployment_recipe.json", {"science": 337})
                else:
                    rows = pd.read_csv(new / "local_output/submission.csv")
                    rows.loc[399, "rank1"] = "CCC"
                    rows.to_csv(new / "local_output/submission.csv", index=False)
                before = c.read()
                with (
                    patch("casmi_ml.one_time_kaggle.check_release", return_value="two"),
                    self.assertRaises(ValueError),
                ):
                    amend_unsubmitted(new, c)
                self.assertEqual(c.read(), before)

    def setup_reserved(self, c, release, status="platform_verified"):
        value = {
            "identity": "one",
            "release": str(release),
            "status": status,
            "remote": {"kernel": "owner/one", "version": 1},
            "message": "unique",
        }
        c.change(lambda state: state.update(one_time_kaggle=value))
        return value

    def test_quota_and_other_pending_prevent_reservation(self):
        with tempfile.TemporaryDirectory() as directory:
            c, release = self.fixture(Path(directory))
            value = self.setup_reserved(c, release)
            c.change(lambda s: s.update(submit_after="2999-01-01T00:00:00Z"))
            self.assertFalse(reserve(c, value))
            c.change(lambda s: s.update(submit_after=None, pending_submission="other"))
            self.assertFalse(reserve(c, value))
            self.assertEqual(c.read()["submissions"], {})

    def test_intent_cannot_be_reserved_or_sent_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            c, release = self.fixture(Path(directory))
            value = self.setup_reserved(c, release)
            self.assertTrue(reserve(c, value))
            self.assertTrue(c.local_only())
            self.assertEqual(c.read()["pending_submission"], "one")
            with self.assertRaisesRegex(ValueError, "Unused explicit"):
                reserve(c, value)
            api = Mock()
            api.competition_submissions.return_value = []
            with patch("casmi_ml.one_time_kaggle.check_release", return_value="one"):
                self.assertEqual(step(c, api), "submission_intent")
            api.competition_submit_code.assert_not_called()
            self.assertTrue(c.local_only())

    def test_accepted_submission_reconciles_then_publisher_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            c, release = self.fixture(Path(directory))
            value = self.setup_reserved(c, release)
            reserve(c, value)
            api = Mock()
            api.competition_submissions.return_value = [
                Mock(
                    description="unique", ref=123, status="COMPLETE", public_score=0.175
                )
            ]
            with (
                patch("casmi_ml.one_time_kaggle.check_release", return_value="one"),
                patch("casmi_ml.one_time_kaggle.update") as update,
            ):
                self.assertEqual(step(c, api), "COMPLETE")
                update.assert_called_once_with(
                    c, status="COMPLETE", submission_id=123, public_score=0.175
                )
            c.change(
                lambda s: s["one_time_kaggle"].update(
                    status="COMPLETE", submission_id=123
                )
            )
            with patch("casmi_ml.one_time_kaggle.check_release", return_value="one"):
                self.assertEqual(step(c, api), "COMPLETE")
            api.competition_submit_code.assert_not_called()
            self.assertIsNone(c.read()["pending_submission"])
            self.assertTrue(c.local_only())
            self.assertEqual(c.read()["public_best"]["score"], 0.176)

    def test_pending_empty_score_reconciles_without_resubmitting(self):
        with tempfile.TemporaryDirectory() as directory:
            c, release = self.fixture(Path(directory))
            value = self.setup_reserved(c, release)
            reserve(c, value)
            api = Mock()
            api.competition_submissions.return_value = [
                Mock(description="unique", ref=124, status="PENDING", public_score="")
            ]
            with (
                patch("casmi_ml.one_time_kaggle.check_release", return_value="one"),
                patch("casmi_ml.one_time_kaggle.update") as receipt,
            ):
                self.assertEqual(step(c, api), "PENDING")
                receipt.assert_called_once_with(
                    c, status="PENDING", submission_id=124, public_score=None
                )
            self.assertEqual(c.read()["pending_submission"], "one")
            api.competition_submit_code.assert_not_called()

    def test_ambiguous_dataset_or_kernel_post_is_never_retried(self):
        with tempfile.TemporaryDirectory() as directory:
            c, release = self.fixture(Path(directory))
            api = Mock()
            for status in ("upload_intent", "kernel_intent"):
                self.setup_reserved(c, release, status)
                with patch(
                    "casmi_ml.one_time_kaggle.check_release", return_value="one"
                ):
                    self.assertEqual(step(c, api), status)
            api.dataset_create_new.assert_not_called()
            api.kernels_push.assert_not_called()


if __name__ == "__main__":
    unittest.main()
