import unittest

from casmi_ml.chembl_high_joint_evidence import supported_promotion


class JointOriginalEvidenceTests(unittest.TestCase):
    def test_fragment_and_reference_positions_both_protected(self):
        prior, current = ["a", "b", "c", "d"], ["a", "b", "c", "x", "d"]
        self.assertEqual(
            supported_promotion(
                prior, current, ["x"], {"a": 1, "b": 3, "c": 2}, {"x": 4}, False, {"b"}
            ),
            (["a", "b", "x", "c", "d"], True),
        )
        self.assertEqual(
            supported_promotion(
                prior, current, ["x"], {"a": 1, "b": 3, "c": 2}, {"x": 2}, False, set()
            ),
            (current, False),
        )
        self.assertEqual(
            supported_promotion(
                prior, current, ["x"], {"a": 1, "b": 3, "c": 2}, {"x": 4}, False, {"c"}
            ),
            (current, True),
        )
