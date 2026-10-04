import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

from casmi_ml import portable_overlap_pipeline as runtime


class SharedPoolConsumerTest(unittest.TestCase):
    def test_final_consumer_uses_early_pool_without_creating_more_workers(self):
        with tempfile.TemporaryDirectory() as root:
            recipe = Path(root) / "recipe.json"
            output = Path(root) / "submission.csv"
            recipe.write_text(
                json.dumps(
                    {
                        "generation": {"total_seconds": 1800},
                        "chemistry": {"fragment_seconds": 1200},
                        "external_routed": {},
                    }
                )
            )
            pool = Mock()
            pool.deadline = 1200
            pool.fetch.return_value = {
                "status": "complete",
                "scores": {"candidate": 1.0},
            }
            stack = Mock()
            frame = SimpleNamespace(
                f_back=SimpleNamespace(
                    f_code=SimpleNamespace(co_name="predict"),
                    f_globals={"__name__": "casmi_ml.research_pipeline"},
                    f_locals={"started": 0},
                )
            )

            def consume(*args, **kwargs):
                from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag

                engine = object.__new__(PersistentMonomerMetFrag)
                result = engine._run(
                    "path", {"candidate": "CCO"}, [(1, 1)], 2, "[M+H]+"
                )
                Path(str(output) + ".report.json").write_text("{}")
                return result

            def pipeline(*args):
                from casmi_ml.portable_streaming_pipeline import extend_streamed
                from casmi_ml.research_pipeline import generate_predict

                generate_predict("checkpoint")
                return extend_streamed(fragment_deadline=1200)

            with (
                patch.object(runtime.inspect, "currentframe", return_value=frame),
                patch.object(
                    runtime,
                    "generate_with_physical_planning",
                    return_value=(None, pool, stack, {}),
                ),
                patch.object(runtime, "original_run", new=pipeline),
                patch("casmi_ml.portable_fragment_prefetch.serial_extend", new=consume),
                patch(
                    "casmi_ml.portable_fragment_prefetch.FragmentPrefetch",
                    side_effect=AssertionError("must not create second pool"),
                ),
            ):
                result = runtime.run(recipe, root, "catalog", output)
            self.assertEqual(
                result, {"status": "complete", "scores": {"candidate": 1.0}}
            )
            pool.fetch.assert_called_once_with(
                "path", {"candidate": "CCO"}, [(1, 1)], 2, "[M+H]+"
            )
            pool.close.assert_called_once()
            stack.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
