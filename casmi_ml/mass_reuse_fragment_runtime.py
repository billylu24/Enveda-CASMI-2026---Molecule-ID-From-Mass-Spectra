"""Bound private peak-loop mass reuse without changing original recipe or requests."""

from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch

from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentRunner


@contextmanager
def bound_mass_reuse(classes, expected):
    classes = Path(classes).absolute()
    actual = {str(p.relative_to(classes)): digest(p) for p in classes.rglob("*.class")}
    required = {
        "MetFragWorker.class",
        "de/ipbhalle/metfraglib/process/CombinedMetFragProcess.class",
        "de/ipbhalle/metfraglib/fragment/DefaultBitArrayFragment.class",
        "de/ipbhalle/metfraglib/fragmenterassignerscorer/TopDownFragmenterAssignerScorer.class",
    }
    if actual != expected or set(actual) != required:
        raise ValueError("Complete bound mass-reuse classes required")
    original = PersistentRunner.__init__

    def initialize(runner, *args, **kwargs):
        original(runner, *args, **kwargs)
        command = runner.command
        if (
            len(command) != 5
            or command[1:3] != ["-Xmx1g", "-cp"]
            or command[4] != "MetFragWorker"
        ):
            raise ValueError("Original JVM heap and worker invocation required")
        command[3] = str(classes) + ":" + command[3]

    with patch.object(PersistentRunner, "__init__", new=initialize):
        yield
