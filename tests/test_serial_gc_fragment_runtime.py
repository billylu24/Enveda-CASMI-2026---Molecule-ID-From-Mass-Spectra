import unittest
from unittest.mock import patch

from casmi_ml import serial_gc_fragment_runtime as runtime
from casmi_ml.metfrag_persistent import PersistentRunner


class SerialGcRuntimeTest(unittest.TestCase):
    def test_java_collector_flag_applies_only_to_child_start_and_restores_classpath_binding(
        self,
    ):
        command = ["java", "-Xmx1g", "-cp", "classes:jar", "MetFragWorker"]
        runner = type("Runner", (), {"command": command})()
        seen = []

        def start(obj):
            seen.extend(obj.command)

        def extend(*args, **kwargs):
            PersistentRunner._start(runner)

        with (
            patch.object(PersistentRunner, "_start", new=start),
            patch.object(runtime, "extend_balanced", new=extend),
        ):
            runtime.extend_serial_gc()
        self.assertEqual(
            seen,
            [
                "java",
                "-Xmx1g",
                "-XX:+UseSerialGC",
                "-cp",
                "classes:jar",
                "MetFragWorker",
            ],
        )
        self.assertIs(runner.command, command)
        self.assertEqual(runner.command[3], "classes:jar")

    def test_failed_child_start_restores_original_command(self):
        command = ["java", "-Xmx1g", "-cp", "classes:jar", "MetFragWorker"]
        runner = type("Runner", (), {"command": command})()

        def start(obj):
            raise RuntimeError("child failure")

        def extend(*args, **kwargs):
            PersistentRunner._start(runner)

        with (
            patch.object(PersistentRunner, "_start", new=start),
            patch.object(runtime, "extend_balanced", new=extend),
            self.assertRaisesRegex(RuntimeError, "child failure"),
        ):
            runtime.extend_serial_gc()
        self.assertIs(runner.command, command)
