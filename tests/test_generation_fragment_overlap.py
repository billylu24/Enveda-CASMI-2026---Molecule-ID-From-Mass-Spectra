import json
import tempfile
import unittest
from pathlib import Path

from casmi_ml.generation_fragment_overlap import CompletedHandoffs


class CompletedHandoffsTest(unittest.TestCase):
    def test_atomic_events_collected_once_and_arrival_order_does_not_change_identity(
        self,
    ):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            collector = CompletedHandoffs()
            collector.add(path)
            (path / "000001.json").write_text(
                json.dumps({"molecule_id": "b", "smiles": ["CC"]})
            )
            (path / "000000.pending").write_text("incomplete")
            collector.collect()
            self.assertEqual(list(collector.rows), ["b"])
            (path / "000000.json").write_text(
                json.dumps({"molecule_id": "a", "smiles": ["CO"]})
            )
            collector.collect()
            collector.collect()
            self.assertEqual(collector.queue.qsize(), 2)
            self.assertEqual(
                collector.rows,
                {
                    "a": {"molecule_id": "a", "smiles": ["CO"]},
                    "b": {"molecule_id": "b", "smiles": ["CC"]},
                },
            )

    def test_duplicate_molecule_from_another_worker_fails_before_planning(self):
        with tempfile.TemporaryDirectory() as root:
            path = Path(root)
            a = path / "a"
            b = path / "b"
            a.mkdir()
            b.mkdir()
            collector = CompletedHandoffs()
            collector.add(a)
            collector.add(b)
            for folder in [a, b]:
                (folder / "000000.json").write_text(
                    json.dumps({"molecule_id": "duplicate", "smiles": ["CCO"]})
                )
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                collector.collect()


if __name__ == "__main__":
    unittest.main()
