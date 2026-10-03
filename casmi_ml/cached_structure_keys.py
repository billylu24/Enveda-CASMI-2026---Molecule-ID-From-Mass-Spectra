"""Bounded pure SMILES identity memoization retaining actual structure semantics."""

from functools import lru_cache

from rdkit import Chem


@lru_cache(maxsize=100000)
def structure_key(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Invalid current structure")
    return Chem.MolToInchiKey(molecule)[:14]
