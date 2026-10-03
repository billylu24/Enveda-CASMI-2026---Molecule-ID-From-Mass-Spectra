import unittest

import numpy as np

from casmi_ml.diagnostic_feature_runtime import maximum_evidence_features
from casmi_ml.diagnostic_residual import legacy_evidence_features
from tests.test_chemistry import spectrum


class MaximumEvidenceRuntimeTests(unittest.TestCase):
    def test_exact_channels_across_random_spectra_modes_and_tolerances(self):
        rng = np.random.default_rng(42)
        for precise in (False, True):
            for i in range(24):
                masses = np.concatenate(
                    [
                        rng.uniform(40, 350, 200),
                        [100, 118.010565, 127.994915, 146.00548, 163.0601],
                    ]
                )
                row = spectrum(masses, precursor=350)
                row["ms2_normalized_intensities"] = rng.uniform(
                    0, 1, len(masses)
                ).tolist()
                if i % 3 == 0:
                    row.update(adduct="[M-H]-", ionization_mode="negative")
                np.testing.assert_array_equal(
                    maximum_evidence_features(row, precise),
                    legacy_evidence_features(row, precise),
                )

    def test_unsupported_and_corrupted_measurements_preserve_empty_evidence(self):
        for row in [
            spectrum([float("nan"), 0, 100]),
            dict(spectrum([100]), instrument_type="unknown"),
            dict(spectrum([100]), adduct="[2M+H]+"),
        ]:
            for precise in (True, False):
                np.testing.assert_array_equal(
                    maximum_evidence_features(row, precise),
                    legacy_evidence_features(row, precise),
                )
