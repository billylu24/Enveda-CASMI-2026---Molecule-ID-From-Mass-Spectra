import threading
import unittest
from pathlib import Path
from types import SimpleNamespace

from casmi_ml.streaming_fragment_prefetch import StreamingFragmentPrefetch


class StreamingFragmentPrefetchTest(unittest.TestCase):
    def test_discovery_starts_bounded_requests_without_restarting(self):
        gate = threading.Event()
        calls = []

        class Prefetch(StreamingFragmentPrefetch):
            def perform(self, request):
                calls.append(str(request[0]))
                gate.wait(3)
                return {"status": "complete", "scores": {"x": 2.0}}

        engine = SimpleNamespace(
            jar="jar",
            cache="cache",
            runner=SimpleNamespace(command=["java", "heap", "-cp", "classes:jar"]),
            java="java",
            threads=2,
            timeout=60,
            depth=2,
        )
        pool = Prefetch(None)
        try:
            for i in range(3):
                pool.record(
                    engine, Path(str(i)), {"x": "CC"}, [(30.0, 1.0)], 30.0, "[M+H]+"
                )
            self.assertEqual(len(pool.futures), 2)
            self.assertEqual(list(pool.waiting), ["2"])
            existing = dict(pool.futures)
            pool.start()
            self.assertEqual(pool.futures, existing)
            gate.set()
            self.assertEqual(
                pool.fetch(Path("0"), {}, [], 0, ""),
                {"status": "complete", "scores": {"x": 2.0}},
            )
            self.assertEqual(pool.fetch(Path("2"), {}, [], 0, "")["scores"], {"x": 2.0})
            self.assertEqual(sorted(calls), ["0", "1", "2"])
        finally:
            gate.set()
            pool.close()
