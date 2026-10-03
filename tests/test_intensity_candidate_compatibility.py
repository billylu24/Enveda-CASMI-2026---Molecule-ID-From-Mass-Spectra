import unittest
from unittest.mock import patch

import numpy as np

from casmi_ml.intensity_candidate_compatibility import evidence_features


class IntensityCandidateTests(unittest.TestCase):
    def test_squares_per_spectrum_observations_before_mean_and_absence_remains_zero(
        self,
    ):
        old = np.linspace(0, 1, 24, dtype=np.float32)
        new = np.linspace(0, 1, 36, dtype=np.float32)
        with (
            patch(
                "casmi_ml.intensity_candidate_compatibility.original_evidence",
                return_value=old,
            ),
            patch(
                "casmi_ml.intensity_candidate_compatibility.extended_evidence",
                return_value=new,
            ),
        ):
            actual = evidence_features({})
        combined = np.concatenate([old, new])
        np.testing.assert_array_equal(actual, combined * combined)
        self.assertTrue(((actual >= 0) & (actual <= 1)).all())
        self.assertEqual(actual[0], 0)
        self.assertEqual(actual[-1], 1)
        with self.assertRaisesRegex(ValueError, "precise"):
            evidence_features({}, False)
