"""Portable exact fingerprint/object storage for the unchanged scientific recipe."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from casmi_ml.metfrag import digest
from casmi_ml.packed_candidate_cache import PackedCandidateIndex
from casmi_ml.portable_verified_pipeline import run as verified_run


class PortableObjectPackedCandidateIndex(PackedCandidateIndex):
    """Convert pandas extension strings without version-sensitive selector syntax."""

    def __init__(self, catalog, store, catalog_sha256):
        super().__init__(catalog, store, catalog_sha256)
        for column, dtype in self.catalog.dtypes.items():
            if isinstance(dtype, pd.StringDtype):
                self.catalog[column] = self.catalog[column].astype(object)


def run(recipe, data_dir, coconut, output):
    recipe = Path(recipe)
    proof = json.loads((recipe.parent / "release_evidence.json").read_text())
    runtime = proof["portable_packed_runtime"]
    for field, source in [
        ("packed_source_sha256", "casmi_ml/packed_candidate_cache.py"),
        ("object_source_sha256", "casmi_ml/object_candidate_runtime.py"),
    ]:
        if runtime[field] != digest(source):
            raise ValueError("Portable packed runtime source changed")
    if runtime["full_rank_matches"] != {"unknown": 2000, "known": 1340}:
        raise ValueError("Exact full local packed/object source controls required")
    config = json.loads(recipe.read_text())["external_routed"]
    factory = lambda catalog: PortableObjectPackedCandidateIndex(
        catalog, recipe.parent / "packed_fingerprints", config["catalog_sha256"]
    )
    with patch("casmi_ml.chembl_routed_inference.CandidateIndex", new=factory):
        return verified_run(recipe, data_dir, coconut, output)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("recipe", "data-dir", "coconut", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    run(**vars(p.parse_args()))


if __name__ == "__main__":
    main()
