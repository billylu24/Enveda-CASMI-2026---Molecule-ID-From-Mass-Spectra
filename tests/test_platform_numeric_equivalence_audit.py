import copy
import math
import unittest

from casmi_ml.platform_numeric_equivalence_audit import compare_dictionaries


class NumericEquivalenceAuditTests(unittest.TestCase):
    def fixture(self):
        return {
            str(i): {
                "status": "complete",
                "jar_sha256": "same",
                "scores": {"a": 100.0, "b": 90.0},
            }
            for i in range(265)
        }

    def test_exact_and_explicit_two_ulp_review(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        b["0"]["scores"]["a"] = math.nextafter(
            math.nextafter(100.0, math.inf), math.inf
        )
        with self.assertRaises(ValueError):
            compare_dictionaries(a, b)
        r = compare_dictionaries(a, b, max_ulp=2, max_relative=1e-15)
        self.assertEqual(r["different_values"], 1)
        self.assertEqual(r["max_ulp_distance"], 2)
        self.assertFalse(r["publisher_authorization"])
        self.assertTrue(compare_dictionaries(a, a)["all_raw_dictionaries_exact"])

    def test_larger_difference_and_nonfinite_rejected(self):
        for v in [
            math.nextafter(
                math.nextafter(math.nextafter(100.0, math.inf), math.inf), math.inf
            ),
            float("nan"),
            float("inf"),
            -100.0,
            "100.0",
            True,
        ]:
            a = self.fixture()
            b = copy.deepcopy(a)
            b["0"]["scores"]["a"] = v
            with self.subTest(v=v), self.assertRaises(ValueError):
                compare_dictionaries(a, b, max_ulp=2, max_relative=1e-15)

    def test_request_candidate_metadata_and_completion_changes_rejected(self):
        for change in ["request", "candidate", "jar", "incomplete"]:
            a = self.fixture()
            b = copy.deepcopy(a)
            if change == "request":
                b.pop("0")
            elif change == "candidate":
                b["0"]["scores"].pop("a")
            elif change == "jar":
                b["0"]["jar_sha256"] = "other"
            else:
                b["0"]["status"] = "timeout"
            with self.subTest(change=change), self.assertRaises(ValueError):
                compare_dictionaries(a, b, max_ulp=2, max_relative=1e-15)

    def test_two_ulp_near_zero_cannot_pass_relative_bound(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        a["0"]["scores"]["a"] = 0.0
        b["0"]["scores"]["a"] = math.nextafter(0.0, math.inf)
        with self.assertRaises(ValueError):
            compare_dictionaries(a, b, max_ulp=2, max_relative=1e-15)

    def test_two_ulp_rank_change_rejected(self):
        a = self.fixture()
        b = copy.deepcopy(a)
        a["0"]["scores"]["b"] = 100.0
        b["0"]["scores"]["b"] = math.nextafter(100.0, math.inf)
        with self.assertRaisesRegex(ValueError, "order changed"):
            compare_dictionaries(a, b, max_ulp=2, max_relative=1e-15)
