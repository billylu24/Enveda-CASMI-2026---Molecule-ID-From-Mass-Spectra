import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_cache import PackedCandidateIndex, prepare
from casmi_ml.ranking import CandidateIndex


class PackedCandidateTests(unittest.TestCase):
    def fixture(self, root):
        source = root / "catalog.parquet"
        cache = root / "cache"
        frame = pd.DataFrame(
            {
                "inchikey14": ["z", "a", "x", "b"],
                "normalized_smiles": ["CCC", "CCO", "broken", "CO"],
                "mass": [44.0, 46.0, 30.0, 32.0],
            }
        )
        frame.to_parquet(source, index=False)
        prepare(source, cache)
        return frame, source, cache

    def test_every_bit_and_candidate_order_match_original_with_invalid_omission(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, source, cache = self.fixture(Path(directory))
            legacy = CandidateIndex(frame)
            packed = PackedCandidateIndex(frame, cache, digest(source))
            for indexes in [[3, 0, 2, 1], [1], [0], []]:
                pool = legacy.catalog.iloc[indexes]
                a, af = legacy.fps(pool)
                b, bf = packed.fps(pool)
                pd.testing.assert_frame_equal(a, b)
                np.testing.assert_array_equal(af, bf)
                self.assertEqual(bf.dtype, np.float32)

    def test_stale_or_tampered_assets_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            frame, source, cache = self.fixture(Path(directory))
            changed = frame.copy()
            changed.loc[0, "normalized_smiles"] = "CCN"
            with self.assertRaisesRegex(ValueError, "identity or ordering"):
                PackedCandidateIndex(changed, cache, digest(source))
            with self.assertRaisesRegex(ValueError, "source binding"):
                PackedCandidateIndex(frame, cache, "wrong")
            data = cache / "fingerprints.npy"
            data.write_bytes(data.read_bytes() + b"changed")
            with self.assertRaisesRegex(ValueError, "cache changed"):
                PackedCandidateIndex(frame, cache, digest(source))
