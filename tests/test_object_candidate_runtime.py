import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.metfrag import digest
from casmi_ml.object_candidate_runtime import ObjectPackedCandidateIndex
from casmi_ml.packed_candidate_cache import PackedCandidateIndex, prepare


class ObjectCandidateRuntimeTests(unittest.TestCase):
    def test_values_order_mass_window_and_fingerprints_survive_arrow_conversion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            frame = pd.DataFrame(
                {
                    "inchikey14": pd.Series(["b", "a", "c"], dtype="str"),
                    "normalized_smiles": pd.Series(["CCN", "CCO", "CCC"], dtype="str"),
                    "mass": [45.0, 46.0, 44.0],
                }
            )
            source = root / "catalog.parquet"
            frame.to_parquet(source, index=False)
            store = root / "packed"
            prepare(source, store)
            base = PackedCandidateIndex(frame, store, digest(source))
            trial = ObjectPackedCandidateIndex(frame, store, digest(source))
            self.assertEqual(trial.catalog.normalized_smiles.dtype, np.dtype("object"))
            for candidates in [
                base.catalog,
                base.catalog.iloc[[1, 0]],
                base.catalog.iloc[:0],
                base.query(45),
            ]:
                a, afp = base.fps(candidates)
                b, bfp = trial.fps(candidates)
                pd.testing.assert_frame_equal(a, b, check_dtype=False)
                np.testing.assert_array_equal(afp, bfp)
            with self.assertRaisesRegex(ValueError, "row identity"):
                changed = frame.copy()
                changed.loc[0, "normalized_smiles"] = "CCCC"
                ObjectPackedCandidateIndex(changed, store, digest(source))
