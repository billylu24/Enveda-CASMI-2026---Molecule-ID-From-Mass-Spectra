import unittest

from casmi_ml.object_candidate_activation import validate_reports


class ObjectCandidateActivationTests(unittest.TestCase):
    def test_budgets_new_failures_and_partial_rank_or_proposal_controls_rejected(self):
        full = {
            "verified": True,
            "development_budget_fallbacks": 0,
            "incremental_new_failure_contexts": 0,
            "unknown": {"full_rank_matches": 2000},
            "known": {"full_rank_matches": 1340},
        }
        proposals = {
            "runtime_verified": True,
            "runtime_complete_proposal_matches": dict.fromkeys(
                ["native", "corrected_only", "corrected_fusion"], 2000
            ),
        }
        validate_reports(full, proposals)
        for key in ["development_budget_fallbacks", "incremental_new_failure_contexts"]:
            with self.assertRaisesRegex(ValueError, "Complete exact"):
                validate_reports({**full, key: 1}, proposals)
        with self.assertRaisesRegex(ValueError, "All3340"):
            validate_reports({**full, "known": {"full_rank_matches": 1339}}, proposals)
        with self.assertRaisesRegex(ValueError, "complete2000"):
            validate_reports(
                full,
                {**proposals, "runtime_complete_proposal_matches": {"native": 2000}},
            )
