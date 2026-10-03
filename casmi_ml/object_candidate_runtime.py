"""Trial candidate frame storage conversion preserving every value and fingerprint."""

from casmi_ml.packed_candidate_cache import PackedCandidateIndex


class ObjectPackedCandidateIndex(PackedCandidateIndex):
    def __init__(self, catalog, store, catalog_sha256):
        super().__init__(catalog, store, catalog_sha256)
        # Validate the original source and sorted row mapping before conversion.
        # Values, indexes, mass order, validity and packed bits stay unchanged.
        for column in self.catalog.select_dtypes(include=["str", "string"]).columns:
            self.catalog[column] = self.catalog[column].astype(object)
