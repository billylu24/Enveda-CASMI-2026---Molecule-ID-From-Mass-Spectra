"""Original balanced requests with one serial collector per1GiB physical JVM."""

from unittest.mock import patch

from casmi_ml.balanced_cpu_fragment_runtime import extend_balanced
from casmi_ml.metfrag_persistent import PersistentRunner


def extend_serial_gc(*args, **kwargs):
    original = PersistentRunner._start

    def start(runner):
        command = runner.command
        if command[1] != "-Xmx1g" or command[2] != "-cp":
            raise ValueError("Original1GiB physical JVM command required")
        runner.command = [*command[:2], "-XX:+UseSerialGC", *command[2:]]
        try:
            return original(runner)
        finally:
            runner.command = command

    with patch.object(PersistentRunner, "_start", new=start):
        return extend_balanced(*args, **kwargs)
