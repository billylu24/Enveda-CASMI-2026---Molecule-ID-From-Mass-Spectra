import json
import tempfile
import threading
import time
import unittest
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.research_loop import Controller


class LaunchOwnershipTests(unittest.TestCase):
    def test_primary_and_auxiliary_share_launch_lock_even_without_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(Path("configs/research_loop.json").read_text())
            config["root"] = str(root / "state")
            write_json(root / "config.json", config)
            controller = Controller(root / "config.json")
            result = []
            first = threading.Thread(
                target=lambda: result.append(
                    controller.job(
                        ["sleep", "10"],
                        root / "round/execution.log",
                        auxiliary_key="owner",
                    )
                )
            )
            first.start()
            deadline = time.monotonic() + 5
            while (
                not controller.read().get("auxiliary_jobs")
                and time.monotonic() < deadline
            ):
                time.sleep(0.01)
            before = controller.read()["auxiliary_jobs"]
            self.assertTrue(before)
            self.assertEqual(
                controller.job(["sleep", "10"], root / "round/other.log"),
                "orphan_job_running",
            )
            self.assertEqual(controller.read()["auxiliary_jobs"], before)
            self.assertIsNone(controller.read()["active_job"])
            controller.stop()
            first.join(timeout=5)
            self.assertEqual(result, ["stopped"])
