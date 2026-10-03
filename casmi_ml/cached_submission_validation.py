"""Memoize pure per-SMILES validation while preserving submission checks and errors."""

from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=100000)
def checked_structure_key(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        return None
    return Chem.MolToInchiKey(molecule)[:14]


def validate_submission(test, submission):
    expected = set(test.molecule_id)
    if (
        set(submission.molecule_id) != expected
        or submission.molecule_id.duplicated().any()
    ):
        raise ValueError("Submission molecule IDs do not match the test set")
    for row in submission.itertuples():
        values = row.smiles.split(";") if isinstance(row.smiles, str) else []
        if not 1 <= len(values) <= 25:
            raise ValueError(f"Invalid candidate count: {row.molecule_id}")
        keys = []
        for smiles in values:
            key = checked_structure_key(smiles)
            if key is None:
                raise ValueError(f"Invalid SMILES for {row.molecule_id}")
            keys.append(key)
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate 2D structures for {row.molecule_id}")
