import unittest
from unittest.mock import patch

import pandas as pd

from casmi_ml.parallel_generation_runtime import ordered_merge, partitions


class ParallelGenerationRuntimeTest(unittest.TestCase):
    def test_worker_cli_passes_the_config_path_to_original_worker(self):
        from casmi_ml.parallel_generation_runtime import main

        with (
            patch(
                "sys.argv", ["parallel_generation_runtime", "--worker", "config.json"]
            ),
            patch("casmi_ml.parallel_generation_runtime.run_worker") as worker,
        ):
            main()
            worker.assert_called_once_with("config.json")

    def test_all_measured_rows_of_each_molecule_stay_together(self):
        frame = pd.DataFrame({"molecule_id": ["c", "a", "c", "b", "a", "d"]})
        shards = partitions(frame)
        self.assertEqual(shards, [["c", "b"], ["a", "d"]])
        self.assertFalse(set(shards[0]) & set(shards[1]))
        self.assertEqual(
            {key for shard in shards for key in shard}, set(frame.molecule_id)
        )

    def test_merge_restores_original_order_and_rejects_missing_or_duplicates(self):
        parts = [
            pd.DataFrame({"molecule_id": ["a", "c"], "smiles": ["C", "CCC"]}),
            pd.DataFrame({"molecule_id": ["b"], "smiles": ["CC"]}),
        ]
        self.assertEqual(
            ordered_merge(parts, ["c", "b", "a"]).smiles.tolist(), ["CCC", "CC", "C"]
        )
        with self.assertRaises(ValueError):
            ordered_merge(parts, ["a", "b"])
        with self.assertRaises(ValueError):
            ordered_merge([parts[0], parts[0]], ["a", "c"])
