"""Source-bound packed fingerprints for unchanged catalog candidates."""

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import rdBase

from casmi_ml.data import fingerprint, write_json
from casmi_ml.metfrag import digest
from casmi_ml.ranking import CandidateIndex
from casmi_ml.research_protocol import freeze


def catalog_identity(catalog):
    hashes = pd.util.hash_pandas_object(
        catalog[["inchikey14", "normalized_smiles", "mass"]], index=False
    ).to_numpy(np.uint64)
    return hashlib.sha256(hashes.tobytes()).hexdigest()


def prepare(catalog, output):
    source, output = Path(catalog), Path(output)
    output.mkdir(parents=True, exist_ok=True)
    frame = CandidateIndex(pd.read_parquet(source)).catalog
    protocol = {
        "source_sha256": digest(__file__),
        "catalog_sha256": digest(source),
        "fingerprint_source_sha256": digest("casmi_ml/data.py"),
        "rdkit_version": rdBase.rdkitVersion,
        "sorted_catalog_identity": catalog_identity(frame),
        "molecules": len(frame),
        "bits": 2048,
        "packing": "numpy packbits default big bitorder; one sorted catalog row maps to one256byte fingerprint, invalid rows separately marked",
        "scope": "Runtime cache only; no chemistry, candidate, score, rule or model change",
        "local_only": True,
        "independent_acceptance": False,
    }
    freeze(output / "protocol.json", protocol)
    report = output / "report.json"
    if report.exists():
        return json.loads(report.read_text())
    packed = np.lib.format.open_memmap(
        output / "fingerprints.npy", mode="w+", dtype=np.uint8, shape=(len(frame), 256)
    )
    valid = np.lib.format.open_memmap(
        output / "valid.npy", mode="w+", dtype=np.bool_, shape=(len(frame),)
    )
    started = time.monotonic()
    for index, smiles in enumerate(frame.normalized_smiles):
        value = fingerprint(smiles)
        valid[index] = value is not None
        if value is not None:
            value = np.asarray(value)
            if value.shape != (2048,) or not np.isin(value, [0, 1]).all():
                raise ValueError("Unsupported source fingerprint")
            encoded = np.packbits(value.astype(np.uint8))
            if not np.array_equal(np.unpackbits(encoded), value):
                raise ValueError("Packed fingerprint differs")
            packed[index] = encoded
        else:
            packed[index] = 0
        if (index + 1) % 100000 == 0:
            print(
                "fingerprints",
                index + 1,
                "seconds",
                time.monotonic() - started,
                flush=True,
            )
    packed.flush()
    valid.flush()
    result = {
        "diagnostic_only": True,
        "local_only": True,
        "independent_acceptance": False,
        "molecules": len(frame),
        "valid": int(valid.sum()),
        "invalid": int((~valid).sum()),
        "every_packed_fingerprint_exact": True,
        "seconds": time.monotonic() - started,
        "fingerprints_sha256": digest(output / "fingerprints.npy"),
        "valid_sha256": digest(output / "valid.npy"),
        "source_catalog_sha256": protocol["catalog_sha256"],
        "sorted_catalog_identity": protocol["sorted_catalog_identity"],
    }
    write_json(report, result)
    return result


class PackedCandidateIndex(CandidateIndex):
    def __init__(self, catalog, store, catalog_sha256):
        super().__init__(catalog)
        store = Path(store)
        report = json.loads((store / "report.json").read_text())
        protocol = json.loads((store / "protocol.json").read_text())
        if (
            not report.get("every_packed_fingerprint_exact")
            or protocol["catalog_sha256"] != catalog_sha256
        ):
            raise ValueError("Packed catalog source binding differs")
        if (
            protocol["fingerprint_source_sha256"] != digest("casmi_ml/data.py")
            or protocol["rdkit_version"] != rdBase.rdkitVersion
        ):
            raise ValueError("Fingerprint implementation or RDKit version changed")
        if catalog_identity(self.catalog) != protocol["sorted_catalog_identity"]:
            raise ValueError("Packed catalog row identity or ordering differs")
        for name, field in [
            ("fingerprints.npy", "fingerprints_sha256"),
            ("valid.npy", "valid_sha256"),
        ]:
            if digest(store / name) != report[field]:
                raise ValueError("Packed fingerprint cache changed")
        self.packed = np.load(store / "fingerprints.npy", mmap_mode="r")
        self.valid = np.load(store / "valid.npy", mmap_mode="r")
        if self.packed.shape != (len(self.catalog), 256) or self.valid.shape != (
            len(self.catalog),
        ):
            raise ValueError("Packed cache dimensions differ")
        if self.catalog.inchikey14.duplicated().any():
            raise ValueError("Packed cache requires unique catalog candidate keys")

    def fps(self, candidates):
        indexes = candidates.index.to_numpy(np.int64)
        indexes = indexes[self.valid[indexes]]
        # This retains original catalog order supplied by candidate_window,
        # including union window deduplication and invalid-SMILES omission.
        return self.catalog.loc[indexes], np.unpackbits(
            np.asarray(self.packed[indexes]), axis=1
        ).astype(np.float32).reshape(-1, 2048)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True)
    parser.add_argument("--output", required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
