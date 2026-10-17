from functools import lru_cache
from rdkit import Chem, RDLogger
from rdkit.Chem.MolStandardize import rdMolStandardize
RDLogger.DisableLog('rdApp.*')
_tautomer_canonicalizer = rdMolStandardize.TautomerEnumerator()
def canonicalize_smiles_to_ik14(smiles: str) -> str:
    """Canonicalize SMILES to InChIKey14, or None if parsing fails.

    Stereochemistry is removed *before* the tautomer canonicalization, not left to fall out of
    the InChIKey14 truncation afterwards. Truncating to InChIKey14 discards the stereo block, so
    it looks like it should already make stereochemistry free -- but the canonicalization runs
    first, and `TautomerEnumerator` will not move a double bond whose position a neighbouring
    stereocentre constrains. The same skeleton drawn with and without that stereocentre can
    therefore land on two different canonical tautomers, and different tautomers are different
    connectivity, which is the part of the InChIKey14 that survives truncation. Stripping first
    makes the canonical form depend only on the skeleton, which is what the competition promises.

    Both the answer key and the submission go through this function, so the two sides can never
    be treated asymmetrically.
    """
    if not isinstance(smiles, str):
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        Chem.RemoveStereochemistry(mol)
        canonical_mol = _tautomer_canonicalizer.Canonicalize(mol)
        inchikey = Chem.MolToInchiKey(canonical_mol)
        return inchikey[:14]
    # RDKit's MolSanitizeException family subclasses ValueError; RuntimeError covers the C++
    # layer and TypeError covers Boost.Python argument errors. One participant's malformed
    # SMILES must never take the whole scoring run down, so a failure here is just a non-match.
    except (ValueError, RuntimeError, TypeError):
        return None

def heavy_atom_composition(smiles: str):
    """Heavy-atom element counts (excluding H, ignoring charge) as a hashable key, or None
    if parsing fails.

    Cheap (no tautomer enumeration) -- used to gate the expensive canonicalization. Anything
    that shares an InChIKey14 with an accepted structure has the same heavy-atom skeleton, so
    a heavy-atom-composition mismatch lets us skip canonicalization with no change to any
    score. We deliberately do NOT gate on full molecular formula: protomers / charge variants
    (e.g. a zwitterion C(=O)[O-]...[N+] vs its neutral-acid protomer C(=O)O...[N+]) share an
    InChIKey14 but differ in H count and charge, so a formula gate would wrongly reject them.
    """
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None
    counts = {}
    for atom in mol.GetAtoms():
        sym = atom.GetSymbol()
        if sym != 'H':
            counts[sym] = counts.get(sym, 0) + 1
    return tuple(sorted(counts.items()))

@lru_cache(maxsize=150000)
def metric_identity(smiles):
    key = canonicalize_smiles_to_ik14(smiles)
    return (key, heavy_atom_composition(smiles)) if key else None

def safe_extend(original, extra, limit=25):
    """Preserve first valid metric identities; fill only unoccupied ranks."""
    out, seen = [], set()
    for smi in list(original) + list(extra):
        smi = str(smi).strip()
        identity = metric_identity(smi)
        if identity is not None and identity not in seen:
            seen.add(identity)
            out.append(smi)
            if len(out) == limit:
                break
    return out

def first_ranks(smiles):
    ranks = {}
    for rank, smi in enumerate(smiles, 1):
        identity = metric_identity(smi)
        if identity is not None:
            ranks.setdefault(identity, rank)
    return ranks

def assert_monotone(original, revised):
    old, new = first_ranks(original), first_ranks(revised)
    assert all(identity in new and new[identity] <= rank for identity, rank in old.items())
