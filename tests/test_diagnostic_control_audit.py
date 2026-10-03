import unittest

from casmi_ml.diagnostic_control_audit import normalize_ranking, structure_key


class DiagnosticIdentityAuditTests(unittest.TestCase):
    def test_uses_actual_structure_identity_instead_of_source_claim(self):
        self.assertEqual(
            normalize_ranking(["claimed"], {"claimed": "CCO"}),
            [structure_key("CCO")],
        )

    def test_collapsing_source_claims_must_fail_instead_of_change_slots(self):
        with self.assertRaisesRegex(ValueError, "duplicates"):
            normalize_ranking(["claim1", "claim2"], {"claim1": "CCO", "claim2": "OCC"})

    def test_invalid_representation_must_fail(self):
        with self.assertRaisesRegex(ValueError, "Invalid source"):
            normalize_ranking(["claimed"], {"claimed": "wrong"})
