import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.development_incremental_metfrag import DevelopmentIncrementalMetFrag
from casmi_ml.metfrag import digest
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag


class DevelopmentIncrementalPhysicsTests(unittest.TestCase):
    def fixture(self, root):
        jar = root / "metfrag.jar"
        jar.write_bytes(b"pinned jar")
        classes = root / "classes"
        files = [
            classes / "MetFragWorker.class",
            classes / "de/ipbhalle/metfraglib/process/CombinedMetFragProcess.class",
        ]
        for p in files:
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(p.name.encode())
        benchmark = root / "benchmark"
        benchmark.mkdir()
        (benchmark / "report.json").write_text(
            json.dumps(
                {
                    "verified": True,
                    "counts": {"failures": 0, "exact_dictionaries": 15, "spectra": 15},
                }
            )
        )
        (benchmark / "protocol.json").write_text(
            json.dumps({"engine": {str(p): digest(p) for p in [jar, *files]}})
        )
        engine = DevelopmentIncrementalMetFrag(
            jar, root / "cache", classes=classes, threads=2, benchmark=benchmark
        )
        return engine

    def test_reuses_exact_completed_scores_and_preserves_absent_candidates(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            engine = self.fixture(root)
            seed = engine.cache / "seed.threads2.json"
            seed.write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "scores": {"old": 80.43},
                        "jar_sha256": engine.sha256,
                    }
                )
            )
            args = ([(100.0, 1.0)], 220.0, "[M+H]+")
            engine._run(
                engine.cache / "seed.json", {"old": "CCO", "absent": "C"}, *args
            )
            requested = []

            def fake(path, candidates, *unused):
                requested.append(candidates)
                result = {
                    "status": "complete",
                    "scores": {"new": 220.58},
                    "jar_sha256": engine.sha256,
                }
                path.with_name(path.stem + ".threads2.json").write_text(
                    json.dumps(result)
                )
                return result

            with patch.object(PersistentMonomerMetFrag, "_run", side_effect=fake):
                result = engine._run(
                    engine.cache / "target.json",
                    {"old": "CCO", "absent": "C", "new": "CC"},
                    *args,
                )
            self.assertEqual(requested, [{"new": "CC"}])
            self.assertEqual(result["scores"], {"old": 80.43, "new": 220.58})
            engine.close()

    def test_failure_does_not_write_partial_full_result(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self.fixture(Path(d))
            with patch.object(
                PersistentMonomerMetFrag,
                "_run",
                return_value={"status": "timeout", "scores": {}},
            ):
                result = engine._run(
                    engine.cache / "target.json",
                    {"new": "CC"},
                    [(100.0, 1.0)],
                    220.0,
                    "[M+H]+",
                )
            self.assertEqual(result["status"], "timeout")
            self.assertFalse((engine.cache / "target.threads2.json").exists())
            engine.close()

    def test_read_only_seed_never_launches_a_new_java_request(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self.fixture(Path(d))
            engine.require_existing = True
            with patch.object(
                PersistentMonomerMetFrag,
                "_run",
                side_effect=AssertionError("New Java forbidden"),
            ):
                result = engine._run(
                    engine.cache / "unseen.json",
                    {"new": "CC"},
                    [(100.0, 1.0)],
                    220.0,
                    "[M+H]+",
                )
            self.assertEqual(result["status"], "not_seeded")
            engine.close()
