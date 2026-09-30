import unittest

import numpy as np

from casmi_ml.robust_router_experiment import KEEP, actions, classifier
from casmi_ml.router_features import FEATURES


class RobustRouterTests(unittest.TestCase):
    def test_high_similarity_guard_overrides_low_reliability(self):
        rows = [
            {"features": [0.999], "baseline_action": "H"},
            {"features": [0.7], "baseline_action": "H"},
        ]
        config = {"type": "logistic", "threshold": 0.4, "guard": 0.99}
        self.assertEqual(actions(rows, config, [0.01, 0.01]), ["H", "F"])

    def test_query_count_cannot_change_classifier_predictions(self):
        index = FEATURES.index("query_spectra")
        self.assertNotIn(index, KEEP)
        rng = np.random.default_rng(42)
        x = rng.normal(size=(100, len(FEATURES)))
        y = (x[:, 0] > 0).astype(int)
        model = classifier("logistic").fit(x, y)
        changed = x.copy()
        changed[:, index] = 1000
        np.testing.assert_array_equal(
            model.predict_proba(x), model.predict_proba(changed)
        )


if __name__ == "__main__":
    unittest.main()
