"""Deterministic training-only SMILES enumeration with vocabulary-safe fallback."""

import hashlib

import numpy as np
from rdkit import Chem


def enumerated_target(smiles, molecule_key, epoch, vocabulary, seed=42):
    if epoch < 1:
        raise ValueError("Positive training epoch required")
    original = vocabulary.encode(smiles)
    if original is None:
        raise ValueError("Original target must fit the frozen vocabulary")
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid training target")
    random_seed = (
        int.from_bytes(
            hashlib.sha256(
                f"smiles-enumeration-v1:{seed}:{molecule_key}:{epoch}".encode()
            ).digest()[:4],
            "big",
        )
        or 1
    )
    # Explicit atom permutations avoid the RDKit random writer's global RNG.
    order = np.random.default_rng(random_seed).permutation(mol.GetNumAtoms()).tolist()
    randomized = Chem.MolToSmiles(
        Chem.RenumberAtoms(mol, order), canonical=False, isomericSmiles=True
    )
    tokens = vocabulary.encode(randomized)
    fallback = tokens is None
    return (original if fallback else tokens), {
        "changed": not fallback and randomized != smiles,
        "vocabulary_or_length_fallback": fallback,
    }
