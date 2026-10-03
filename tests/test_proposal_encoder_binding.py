import unittest

from casmi_ml.chembl_critic_slots import validate_proposal_source


class ProposalEncoderSourceTests(unittest.TestCase):
    def test_original_preselection_can_be_scored_by_alternate_critic_encoder(self):
        validate_proposal_source(
            {"catalog_sha256": "catalog", "encoder_sha256": "original"},
            "catalog",
            "original",
            "new",
        )

    def test_explicit_alternate_preselection_requires_matching_scoring_encoder(self):
        validate_proposal_source(
            {
                "catalog_sha256": "catalog",
                "encoder_sha256": "new",
                "alternate_encoder": True,
            },
            "catalog",
            "original",
            "new",
        )
        for protocol in (
            {"catalog_sha256": "catalog", "encoder_sha256": "new"},
            {
                "catalog_sha256": "catalog",
                "encoder_sha256": "other",
                "alternate_encoder": True,
            },
            {"catalog_sha256": "wrong", "encoder_sha256": "original"},
        ):
            with self.assertRaises(ValueError):
                validate_proposal_source(protocol, "catalog", "original", "new")
