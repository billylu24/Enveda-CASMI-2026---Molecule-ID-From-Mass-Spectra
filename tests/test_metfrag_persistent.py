import subprocess
import sys
import unittest

from casmi_ml.metfrag_persistent import PersistentRunner

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
