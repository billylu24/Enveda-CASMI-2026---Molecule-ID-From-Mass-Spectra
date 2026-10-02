"""Conservative atom-mass ceiling; final exact mass validation remains mandatory."""

import re

from rdkit import Chem


def token_atom_masses(vocabulary):
    """Ignore implicit/bracket hydrogen; count atom and isotope tokens only.

    This is a lower bound on a neutral structure's mass, not a formula oracle
    or a complete SMILES grammar. Unknown tokens receive zero mass.
    """
    table = Chem.GetPeriodicTable()
    result = []
    for token in vocabulary.tokens:
        isotope = 0
        if token.startswith("["):
            match = re.match(r"^\[(\d*)([A-Z][a-z]?|[cnospb])", token)
            if match:
                isotope, element = int(match[1] or 0), match[2].capitalize()
            else:
                element = None
        else:
            element = (
                token.capitalize()
                if token
                in [
                    "B",
                    "C",
                    "N",
                    "O",
                    "P",
                    "S",
                    "F",
                    "Cl",
                    "Br",
                    "I",
                    "c",
                    "n",
                    "o",
                    "s",
                ]
                else None
            )
        mass = 0.0
        if element:
            atomic_number = table.GetAtomicNumber(element)
            mass = (
                table.GetMassForIsotope(atomic_number, isotope)
                if isotope
                else table.GetMostCommonIsotopeMass(atomic_number)
            )
        result.append(mass)
    return result
