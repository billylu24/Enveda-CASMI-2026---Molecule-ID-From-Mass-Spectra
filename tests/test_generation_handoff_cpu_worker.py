import unittest
from unittest.mock import Mock, patch

from casmi_ml import generation_handoff_cpu_worker as worker
from casmi_ml.portable_overlap_cpu_pipeline import redirect_worker


class GenerationCPUWorkerTest(unittest.TestCase):
    def test_original_seeding_and_parameters_passed_with_only_threads_changed(self):
        configure = Mock()

        def original(path):
            self.assertEqual(path, "original_worker.json")
            worker.original.configure(seed=731, threads=4)
            return "unchanged"

        with (
            patch.object(worker.original, "configure", new=configure),
            patch.object(worker, "original_worker", side_effect=original),
        ):
            self.assertEqual(worker.run_worker("original_worker.json"), "unchanged")
            self.assertIs(worker.original.configure, configure)
        configure.assert_called_once_with(seed=731, threads=1)

    def test_configure_restored_on_original_exception(self):
        configure = worker.original.configure
        with patch.object(
            worker, "original_worker", side_effect=ValueError("original error")
        ), self.assertRaisesRegex(ValueError, "original error"):
            worker.run_worker("original.json")
        self.assertIs(worker.original.configure, configure)

    def test_only_completed_handoff_worker_command_redirected(self):
        original = [
            "python",
            "-m",
            "casmi_ml.generation_handoff_stream",
            "--worker",
            "config.json",
        ]
        changed = redirect_worker(original)
        self.assertEqual(changed[3:], original[3:])
        self.assertEqual(changed[2], "casmi_ml.generation_handoff_cpu_worker")
        self.assertEqual(original[2], "casmi_ml.generation_handoff_stream")
        java = ["java", "-Xmx1g", "-cp", "classes:jar", "MetFragWorker"]
        self.assertIs(redirect_worker(java), java)
