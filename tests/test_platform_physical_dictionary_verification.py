import tempfile
import unittest
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.one_time_kaggle import verify_physical_dictionaries


class PlatformPhysicalDictionaryTests(unittest.TestCase):
    def fixture(self, root):
        for folder in ("local_output", "kaggle_output"):
            for i in range(265):
                write_json(
                    root
                    / folder
                    / "submission.csv.external_fragment_cache"
                    / f"{i}.threads2.json",
                    {
                        "status": "complete",
                        "scores": {"candidate": 100.0},
                        "jar_sha256": "same",
                    },
                )

    def test_complete_exact_physical_evidence_passes(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            self.fixture(root)
            verify_physical_dictionaries(root)
            self.assertTrue(
                (root / "platform_physical_dictionary_verification.json").is_file()
            )

    def test_missing_request_and_single_ulp_change_are_rejected(self):
        for change in ("missing", "ulp"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = Path(folder)
                self.fixture(root)
                target = (
                    root
                    / "kaggle_output/submission.csv.external_fragment_cache/264.threads2.json"
                )
                if change == "missing":
                    target.unlink()
                else:
                    write_json(
                        target,
                        {
                            "status": "complete",
                            "scores": {"candidate": 100.00000000000001},
                            "jar_sha256": "same",
                        },
                    )
                with self.assertRaisesRegex(ValueError, "All265"):
                    verify_physical_dictionaries(root)
                self.assertFalse(
                    (root / "platform_physical_dictionary_verification.json").exists()
                )
