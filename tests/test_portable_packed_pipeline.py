import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from casmi_ml.portable_packed_pipeline import PortableObjectPackedCandidateIndex


class PortableStorageTests(unittest.TestCase):
    def test_legacy_and_future_string_dtype_values_indexes_remain_exact(self):
        for dtype in (
            object,
            pd.StringDtype(storage="python", na_value=pd.NA),
            pd.StringDtype(storage="python", na_value=np.nan),
        ):
            with self.subTest(dtype=dtype):
                frame = pd.DataFrame(
                    {
                        "smiles": pd.Series(["CC", "CCC"], dtype=dtype),
                        "mass": [30.0, 44.0],
                    }
                )
                frame.index = [5, 8]

                def initialize(index, catalog, store, sha):
                    index.catalog = catalog.copy()

                with patch(
                    "casmi_ml.packed_candidate_cache.PackedCandidateIndex.__init__",
                    new=initialize,
                ):
                    result = PortableObjectPackedCandidateIndex(
                        frame, "unused", "unused"
                    ).catalog
                self.assertEqual(result.smiles.dtype, object)
                self.assertEqual(list(result.smiles), ["CC", "CCC"])
                self.assertEqual(list(result.index), [5, 8])
                np.testing.assert_array_equal(result.mass, frame.mass)


if __name__ == "__main__":
    unittest.main()
