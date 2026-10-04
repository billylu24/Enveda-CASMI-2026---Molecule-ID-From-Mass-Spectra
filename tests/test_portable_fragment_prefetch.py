import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.portable_fragment_prefetch import FragmentPrefetch


class PrefetchTests(unittest.TestCase):
    def test_same_requests_are_reused_and_on_demand_original_pool_completes(self):
        engines = []

        class Dummy:
            def __init__(self, **parameters):
                self.timeout = parameters["timeout"]
                self.calls = []
                self.closed = False
                engines.append(self)

            def _run(self, path, candidates, peaks, mass, adduct):
                self.calls.append(str(path))
                return {
                    "status": "complete",
                    "scores": {k: float(len(s)) for k, s in candidates.items()},
                }

            def close(self):
                self.closed = True

        with (
            tempfile.TemporaryDirectory() as directory,
            patch(
                "casmi_ml.portable_fragment_prefetch.PersistentMonomerMetFrag",
                new=Dummy,
            ),
        ):
            pool = FragmentPrefetch(None)
            pool.parameters = {"timeout": 60}
            for i in range(5):
                path = Path(directory) / str(i)
                pool.requests[str(path)] = (
                    path,
                    {str(i): "CC"},
                    [(1.0, 1.0)],
                    30.0,
                    "[M+H]+",
                )
            try:
                pool.start()
                path = Path(directory) / "original"
                self.assertEqual(
                    pool.fetch(path, {"first": "CCC"}, [(1.0, 1.0)], 44.0, "[M+H]+"),
                    {"status": "complete", "scores": {"first": 3.0}},
                )
                for request in pool.requests.values():
                    self.assertEqual(
                        pool.fetch(*request)["scores"], {next(iter(request[1])): 2.0}
                    )
                self.assertLessEqual(len(engines), 2)
                self.assertEqual(
                    pool.fetch(path, {"first": "CCC"}, [(1.0, 1.0)], 44.0, "[M+H]+"),
                    {"status": "complete", "scores": {"first": 3.0}},
                )
            finally:
                pool.close()
            self.assertTrue(all(engine.closed for engine in engines))
            self.assertEqual(
                sum(engine.calls.count(str(path)) for engine in engines), 1
            )

    def test_expired_scene_does_not_launch_a_worker_or_extend_deadline(self):
        pool = FragmentPrefetch(0)
        pool.parameters = {"timeout": 60}
        request = (Path("unused"), {"x": "CC"}, [(1.0, 1.0)], 30.0, "[M+H]+")
        with patch(
            "casmi_ml.portable_fragment_prefetch.PersistentMonomerMetFrag"
        ) as engine:
            try:
                pool.start()
                self.assertEqual(
                    pool.fetch(*request), {"status": "timeout", "scores": {}}
                )
                engine.assert_not_called()
            finally:
                pool.close()


if __name__ == "__main__":
    unittest.main()
