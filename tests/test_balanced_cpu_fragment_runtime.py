import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from casmi_ml import balanced_cpu_fragment_runtime as runtime
from casmi_ml.data import write_json


class BalancedCpuRuntimeTest(unittest.TestCase):
    def test_same_invocation_index_reuse_and_scoring_threads_restored(self):
        catalog = pd.DataFrame({"x": [1]})
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "result.csv"
            calls = []
            constructed = []

            def read(path):
                calls.append(Path(path).name)
                return (
                    catalog
                    if Path(path).name == "catalog.parquet"
                    else pd.DataFrame({"query": [2]})
                )

            def constructor(frame):
                constructed.append(frame)
                return object()

            def streamed(*args, **kwargs):
                runtime.base.configure(threads=4)
                self.assertEqual(torch.get_num_threads(), 1)
                one = runtime.base.CandidateIndex(
                    runtime.base.pd.read_parquet(root / "catalog.parquet")
                )
                two = runtime.base.CandidateIndex(
                    runtime.base.pd.read_parquet(root / "catalog.parquet")
                )
                self.assertIs(one, two)
                self.assertEqual(
                    runtime.base.pd.read_parquet(root / "test.parquet")[
                        "query"
                    ].tolist(),
                    [2],
                )
                write_json(str(output) + ".external.report.json", {})
                return "complete"

            before = torch.get_num_threads()
            with (
                patch.object(runtime.base.pd, "read_parquet", new=read),
                patch.object(runtime.base, "CandidateIndex", new=constructor),
                patch.object(runtime, "extend_streamed", new=streamed),
            ):
                result = runtime.extend_balanced(
                    None, None, None, None, output, root, {"catalog": "catalog.parquet"}
                )
            self.assertEqual(result, "complete")
            self.assertEqual(calls, ["catalog.parquet", "test.parquet"])
            self.assertEqual(len(constructed), 1)
            self.assertEqual(torch.get_num_threads(), before)

    def test_threads_restore_after_failed_physical_proof(self):
        before = torch.get_num_threads()

        def fail(*args, **kwargs):
            runtime.base.configure(threads=4)
            raise ValueError("physical timeout")

        with (
            patch.object(runtime, "extend_streamed", new=fail),
            self.assertRaisesRegex(ValueError, "physical timeout"),
        ):
            runtime.extend_balanced(
                None,
                None,
                None,
                None,
                "result.csv",
                ".",
                {"catalog": "catalog.parquet"},
            )
        self.assertEqual(torch.get_num_threads(), before)
