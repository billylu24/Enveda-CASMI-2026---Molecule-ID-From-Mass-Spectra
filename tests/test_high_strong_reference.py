import unittest
from unittest.mock import Mock, patch

import pandas as pd

from casmi_ml.chembl_high_strong_reference import strong_reference_keys


class StrongReferenceEvidenceTests(unittest.TestCase):
    def test_same_fixed_threshold_and_max_over_all_observable_mass_centers(self):
        query = pd.DataFrame({"precursor_mz": [100.0]})
        reference = Mock()
        reference.rank.side_effect = [
            [("a", 0.49999), ("b", 0.5), ("c", 0.7)],
            [("a", 0.6), ("c", 0.3)],
        ]
        with patch(
            "casmi_ml.chembl_high_strong_reference.mass_centers",
            return_value=[100.0, 101.0],
        ):
            observed, scores = strong_reference_keys(reference, query)
        self.assertEqual(observed, {"a", "b", "c"})
        self.assertEqual(scores, {"a": 0.6, "b": 0.5, "c": 0.7})
        self.assertEqual(reference.rank.call_count, 2)
