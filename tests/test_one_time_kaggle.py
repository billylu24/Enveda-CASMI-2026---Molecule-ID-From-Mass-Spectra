import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from casmi_ml.data import write_json
from casmi_ml.one_time_kaggle import authorize, reserve, step
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
