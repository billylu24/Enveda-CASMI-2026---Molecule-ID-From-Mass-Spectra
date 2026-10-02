import unittest
from unittest.mock import patch

import pandas as pd
import torch

from casmi_ml.generation_model_mixture import combine, stream_seed


class ModelMixtureTests(unittest.TestCase):
    def test_each_arm_merges_exactly128_trajectories_with_shared_first_half(self):
        parts = [
            (
                torch.full((64, 2), i),
                torch.full((64,), float(i)),
                torch.ones(64, dtype=torch.bool),
            )
            for i in [1, 2, 3]
        ]
        with patch("casmi_ml.generation_model_mixture.validate_generated") as validate:
            combine(parts[:2], None, None, [], [])
            control = validate.call_args.args
            combine([parts[0], parts[2]], None, None, [], [])
            mixture = validate.call_args.args
        for args in [control, mixture]:
            self.assertEqual([len(x) for x in args[:3]], [128, 128, 128])
        self.assertEqual(control[0][:64], mixture[0][:64])
        self.assertNotEqual(control[0][64:], mixture[0][64:])

    def test_stream_identity_ignores_labels_and_row_order(self):
        frame = pd.DataFrame(
            {
                "adduct": ["[M+H]+", "[M-H]-"],
                "precursor_mz": [100.0, 98.0],
                "ms2_mzs": [[40.0], [50.0]],
                "ms2_normalized_intensities": [[1.0], [1.0]],
            }
        )
        changed = frame.iloc[::-1].assign(
            inchikey14="changed", normalized_smiles="changed"
        )
        for stream in [0, 1]:
            self.assertEqual(stream_seed(frame, stream), stream_seed(changed, stream))
        self.assertNotEqual(stream_seed(frame, 0), stream_seed(frame, 1))
        with self.assertRaises(ValueError):
            stream_seed(frame, 2)


if __name__ == "__main__":
    unittest.main()
