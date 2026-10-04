import tempfile
import unittest
from pathlib import Path

from casmi_ml.mass_reuse_fragment_runtime import bound_mass_reuse
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentRunner

REQUIRED = (
    "MetFragWorker.class",
    "de/ipbhalle/metfraglib/process/CombinedMetFragProcess.class",
    "de/ipbhalle/metfraglib/fragment/DefaultBitArrayFragment.class",
    "de/ipbhalle/metfraglib/fragmenterassignerscorer/TopDownFragmenterAssignerScorer.class",
)


class MassReuseFragmentRuntimeTest(unittest.TestCase):
    def build(self, root):
        for name in REQUIRED:
            path = root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(name.encode())
        return {name: digest(root / name) for name in REQUIRED}

    def test_exact_classes_only_add_classpath_and_restore_runner(self):
        original = PersistentRunner.__init__
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = self.build(root)
            with bound_mass_reuse(root, expected):
                runner = PersistentRunner("java", "pinned.jar", "original_classes")
                self.assertEqual(runner.command[0:3], ["java", "-Xmx1g", "-cp"])
                self.assertEqual(
                    runner.command[3],
                    str(root)
                    + ":"
                    + str(Path("original_classes").absolute())
                    + ":pinned.jar",
                )
                self.assertEqual(runner.command[4], "MetFragWorker")
                self.assertIsNone(runner.process)
            self.assertIs(PersistentRunner.__init__, original)

    def test_class_mutation_and_unbound_extra_class_fail_before_runner(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            expected = self.build(root)
            (root / REQUIRED[0]).write_bytes(b"changed")
            with (
                self.assertRaisesRegex(ValueError, "bound mass-reuse"),
                bound_mass_reuse(root, expected),
            ):
                self.fail("unbound mutation accepted")
            expected = self.build(root)
            (root / "Extra.class").write_bytes(b"code")
            with (
                self.assertRaisesRegex(ValueError, "bound mass-reuse"),
                bound_mass_reuse(root, expected),
            ):
                self.fail("unbound extra class accepted")
