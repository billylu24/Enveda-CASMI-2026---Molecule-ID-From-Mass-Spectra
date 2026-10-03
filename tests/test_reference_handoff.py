import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy import sparse

from baseline import make_matrix
from casmi_ml.reference_handoff import save_reference


class PrivateReferenceHandoffTests(unittest.TestCase):
    def test_chunked_csr_preserves_original_every_float_index_and_empty_rows(self):
        records = [
            (100.0, "a", "CCO", {100: 0.123, 20: 0.4}),
            (101.0, "b", "CCN", {}),
        ] * 4500
        original = make_matrix([r[3] for r in records])
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "reference"
            save_reference(records, out)
            actual = sparse.load_npz(out / "spectra.npz")
        self.assertEqual(original.shape, actual.shape)
        for field in ("data", "indices", "indptr"):
            np.testing.assert_array_equal(
                getattr(original, field), getattr(actual, field)
            )
