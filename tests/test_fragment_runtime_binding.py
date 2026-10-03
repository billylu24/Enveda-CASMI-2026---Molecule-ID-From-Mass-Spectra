import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.chembl_routed_inference import runtime_options
from casmi_ml.metfrag_polling_patch import CLASS_NAME, PATCHED_SHA, PROCESS_SHA


class FragmentRuntimeBindingTests(unittest.TestCase):
    def test_old_configuration_has_no_override_and_rejects_unbound_class(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "classes/MetFragWorker.class"
            self.assertEqual(runtime_options(root, {}, worker), {})
            process = worker.parent / CLASS_NAME
            process.parent.mkdir(parents=True)
            process.write_bytes(b"fake")
            with self.assertRaisesRegex(ValueError, "Unbound"):
                runtime_options(root, {}, worker)

    def test_runtime_requires_exact_class_path_checksum_and_frozen_settings(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            worker = root / "classes/MetFragWorker.class"
            runtime = {
                "candidate_threads": 2,
                "polling_ms": 10,
                "process_class": str(Path("classes") / CLASS_NAME),
                "process_class_sha256": PATCHED_SHA,
                "original_process_class_sha256": PROCESS_SHA,
            }
            with patch(
                "casmi_ml.chembl_routed_inference.digest", return_value=PATCHED_SHA
            ):
                self.assertEqual(
                    runtime_options(root, {"runtime": runtime}, worker), {"threads": 2}
                )
                for key, value in [
                    ("candidate_threads", 1),
                    ("polling_ms", 100),
                    ("process_class", "elsewhere"),
                    ("process_class_sha256", "wrong"),
                    ("original_process_class_sha256", "wrong"),
                ]:
                    with self.assertRaises(ValueError):
                        runtime_options(
                            root, {"runtime": dict(runtime, **{key: value})}, worker
                        )
            with (
                patch("casmi_ml.chembl_routed_inference.digest", return_value="wrong"),
                self.assertRaisesRegex(ValueError, "binding"),
            ):
                runtime_options(root, {"runtime": runtime}, worker)
