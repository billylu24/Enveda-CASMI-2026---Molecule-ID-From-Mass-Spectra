import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag, PersistentRunner

FAKE_WORKER = """
import sys,time
for line in sys.stdin:
    path=line.rstrip("\\n")
    if path=="sleep": time.sleep(3)
    if path=="crash": sys.exit(2)
    print("CASMI_METFRAG_DONE\\t"+path,flush=True)
"""


class PersistentRequestLifecycleTests(unittest.TestCase):
    def runner(self, max_requests=100):
        runner = PersistentRunner(
            "unused", "unused", "unused", max_requests=max_requests
        )
        runner.command = [sys.executable, "-u", "-c", FAKE_WORKER]
        self.addCleanup(runner.close)
        return runner

    def test_timeout_terminates_then_restarts_for_next_request(self):
        runner = self.runner()
        self.assertEqual(runner.run("one", 1).returncode, 0)
        pid = runner.process.pid
        with self.assertRaises(subprocess.TimeoutExpired):
            runner.run("sleep", 0.05)
        self.assertIsNone(runner.process)
        self.assertEqual(runner.run("two", 1).returncode, 0)
        self.assertNotEqual(runner.process.pid, pid)

    def test_worker_failure_does_not_leave_a_live_process(self):
        runner = self.runner()
        self.assertNotEqual(runner.run("crash", 1).returncode, 0)
        self.assertIsNone(runner.process)
        self.assertEqual(runner.run("recovery", 1).returncode, 0)

    def test_bounded_request_count_restarts_jvm(self):
        runner = self.runner(max_requests=1)
        self.assertEqual(runner.run("one", 1).returncode, 0)
        pid = runner.process.pid
        self.assertEqual(runner.run("two", 1).returncode, 0)
        self.assertNotEqual(runner.process.pid, pid)


class CandidateThreadConfigurationTests(unittest.TestCase):
    def test_default_preserves_single_thread_and_explicit_threads_are_bounded(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jar = root / "jar"
            jar.write_bytes(b"fake")
            with patch("casmi_ml.metfrag_persistent.PersistentRunner"):
                self.assertEqual(
                    PersistentMonomerMetFrag(
                        jar, root / "default", classes=root
                    ).threads,
                    1,
                )
                self.assertEqual(
                    PersistentMonomerMetFrag(
                        jar, root / "two", classes=root, threads=2
                    ).threads,
                    2,
                )
                for threads in (0, 3, 4, True):
                    with self.assertRaises(ValueError):
                        PersistentMonomerMetFrag(
                            jar, root / "bad", classes=root, threads=threads
                        )

    def test_thread_variant_cannot_read_legacy_score_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jar = root / "jar"
            jar.write_bytes(b"fake")
            legacy = root / "score.json"
            legacy.write_text('{"status":"complete","scores":{"legacy":1}}')
            variant = root / "score.threads2.json"
            variant.write_text('{"status":"complete","scores":{"variant":2}}')
            with patch("casmi_ml.metfrag_persistent.PersistentRunner"):
                engine = PersistentMonomerMetFrag(jar, root, classes=root, threads=2)
                self.assertEqual(
                    engine._run(legacy, {}, [], 0.0, "[M+H]+")["scores"], {"variant": 2}
                )
                unchanged = PersistentMonomerMetFrag(jar, root, classes=root)
                self.assertEqual(
                    unchanged._run(legacy, {}, [], 0.0, "[M+H]+")["scores"],
                    {"legacy": 1},
                )
