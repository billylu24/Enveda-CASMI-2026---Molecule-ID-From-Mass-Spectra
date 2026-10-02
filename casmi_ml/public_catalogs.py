"""Build auditable, offline candidate catalogs from public structure downloads.

No download or API requests are performed here. Import exact downloaded files,
record the public source URL and license, then ship the resulting parquet file
with inference. Molecular graphs, formal charges and tautomers are preserved;
disconnected structures are rejected rather than silently stripping salts.

Examples::

    python -m casmi_ml.public_catalogs import chebi.sdf.gz --source chebi \
      --source-url https://ftp.ebi.ac.uk/pub/databases/chebi/SDF/chebi.sdf.gz \
      --license 'CC BY 4.0' --output external/chebi/structures.parquet
    python -m casmi_ml.public_catalogs import coconut.csv --source coconut \
      --schema configs/coconut-columns.json --source-url PUBLIC_URL \
      --license LICENSE --output external/coconut/structures.parquet
    python -m casmi_ml.public_catalogs merge base.parquet chebi.parquet \
      lipidmaps.parquet --output external/combined.parquet
"""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass, field
import gzip
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Iterator
from urllib.parse import urlparse
import zipfile

import pandas as pd
import rdkit
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


CATALOG_COLUMNS = [
    "inchikey14", "normalized_smiles", "mass", "exact_mass", "molecular_formula",
    "formal_charge", "origin", "source_id", "original_smiles", "raw_inchikey",
    "provenance_json",
]


@dataclass(frozen=True)
class ColumnSchema:
    """Exact tabular column / SDF property names; only structure is required.

    SDF molecules supply their own graph, so a SMILES property is optional.
    For COCONUT (whose releases have different schemas), callers must supply
    this mapping explicitly. A tabular mapping always needs ``smiles``.
    """

    smiles: str | None = None
    source_id: str | None = None
    inchikey: str | None = None
    exact_mass: str | None = None
    formula: str | None = None
    charge: str | None = None


@dataclass(frozen=True)
class SourceSpec:
    source: str
    source_url: str
    license: str
    schema: ColumnSchema | None = None
    release: str | None = None
    expected_sha256: str | None = None
    source_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.source.strip() or not self.license.strip():
            raise ValueError("source name and license are required")
        url = urlparse(self.source_url)
        if url.scheme not in {"http", "https"} or not url.netloc:
            raise ValueError("a caller-supplied public source URL is required")
        if self.expected_sha256 is not None and not re.fullmatch(
            r"[0-9a-fA-F]{64}", self.expected_sha256
        ):
            raise ValueError("expected_sha256 must be a SHA256 hex digest")
        json.dumps(self.source_metadata, allow_nan=False)


# These are release-specific property names, never positional assumptions.
# Override --schema when a download uses another release's headers.
SDF_SCHEMAS = {
    "chebi": ColumnSchema("SMILES", "ChEBI ID", "INCHIKEY", "MONOISOTOPIC_MASS", "FORMULA", "CHARGE"),
    "lipidmaps": ColumnSchema("SMILES", "LM_ID", "INCHI_KEY", "EXACT_MASS", "FORMULA"),
    "pubchem": ColumnSchema(None, "PUBCHEM_COMPOUND_CID", "PUBCHEM_IUPAC_INCHIKEY", "PUBCHEM_EXACT_MASS", "PUBCHEM_MOLECULAR_FORMULA"),
    "pubchemlite": ColumnSchema(None, "Identifier", "FirstBlock", "MonoisotopicMass", "MolecularFormula"),
}
TABULAR_SCHEMAS = {
    "pubchemlite": ColumnSchema("SMILES", "Identifier", "FirstBlock", "MonoisotopicMass", "MolecularFormula"),
}


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _kind(path: Path) -> str:
    name = path.name.lower()
    if name.endswith((".sdf", ".sdf.gz", ".sdf.zip")):
        return "sdf"
    if name.endswith(".zip"):
        return "sdf"
    if name.endswith((".csv", ".csv.gz", ".tsv", ".tsv.gz", ".parquet")):
        return "table"
    raise ValueError("input must be CSV, TSV, parquet, SDF, gzipped SDF or zipped SDF")


def _clean(value) -> str | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    result = str(value).strip()
    return result if result else None


def _properties(mol, schema: ColumnSchema) -> dict:
    values = {
        key: _clean(mol.GetProp(name)) if name and mol.HasProp(name) else None
        for key, name in asdict(schema).items()
    }
    # Biological classifications can support later chemical-prior features.
    # Preserve reported annotations, without turning them into hard labels.
    annotation_names = ["CATEGORY", "MAIN_CLASS", "SUB_CLASS", "ABBREVIATION", "NAME", "ChEBI NAME", "PUBCHEM_CID", "CHEBI_ID", "PubChem Compound Database Links", "LIPID MAPS Database Links"]
    values["annotations"] = {name: mol.GetProp(name) for name in annotation_names if mol.HasProp(name)}
    return values


def _sdf_records(handle, schema: ColumnSchema, member: str | None = None):
    # Forward supplier handles large .gz/zip members without extracting them.
    supplier = Chem.ForwardSDMolSupplier(handle, sanitize=True, removeHs=False)
    for number, mol in enumerate(supplier, 1):
        record = f"{member}:{number}" if member else str(number)
        values = _properties(mol, schema) if mol is not None else {}
        values["structure_source"] = "sdf"
        yield record, values, mol


def _records(path: Path, schema: ColumnSchema, archive_members: list) -> Iterator:
    if _kind(path) == "sdf":
        if path.name.lower().endswith(".zip"):
            with zipfile.ZipFile(path) as archive:
                names = sorted(n for n in archive.namelist() if n.lower().endswith(".sdf"))
                if not names:
                    raise ValueError("zip archive has no SDF member")
                for name in names:
                    digest = hashlib.sha256()
                    with archive.open(name) as handle:
                        for block in iter(lambda: handle.read(1024 * 1024), b""):
                            digest.update(block)
                    archive_members.append({"path": name, "sha256": digest.hexdigest()})
                    with archive.open(name) as handle:
                        yield from _sdf_records(handle, schema, name)
        else:
            opener = gzip.open if path.name.lower().endswith(".gz") else open
            with opener(path, "rb") as handle:
                yield from _sdf_records(handle, schema)
        return
    if not schema.smiles:
        raise ValueError("tabular input requires an explicit SMILES column")
    if path.suffix.lower() == ".parquet":
        chunks = [pd.read_parquet(path)]
    else:
        sep = "\t" if ".tsv" in path.name.lower() else ","
        chunks = pd.read_csv(path, sep=sep, dtype=str, chunksize=25000)
    required = [value for value in asdict(schema).values() if value]
    offset = 0
    for chunk in chunks:
        missing = sorted(set(required) - set(chunk.columns))
        if missing:
            raise ValueError(f"mapped columns absent from input: {missing}")
        for row in chunk.to_dict("records"):
            offset += 1
            values = {key: _clean(row.get(name)) if name else None for key, name in asdict(schema).items()}
            values["structure_source"] = "smiles"
            smiles = values["smiles"]
            mol = Chem.MolFromSmiles(smiles) if smiles else None
            yield str(offset), values, mol


def _formula_atoms(formula: str) -> dict | None:
    """Compare elemental counts, tolerating spacing and terminal charge syntax."""
    formula = re.sub(r"\s+", "", formula)
    formula = re.sub(r"[+-]\d*$", "", formula)
    tokens = re.findall(r"([A-Z][a-z]?)(\d*)", formula)
    if "".join(element + number for element, number in tokens) != formula:
        return None  # Isotope/other notation is retained for audit, not guessed.
    counts = Counter()
    for element, number in tokens:
        counts[element] += int(number or "1")
    return dict(counts)


def _validate(values: dict, mol, mass_tolerance: float):
    if mol is None:
        return None, "invalid_structure"
    if not mol.GetNumAtoms():
        return None, "empty_structure"
    if len(Chem.GetMolFrags(mol)) != 1:
        return None, "disconnected_structure"
    if any(atom.GetAtomicNum() == 0 for atom in mol.GetAtoms()):
        return None, "unsupported_atom"
    canonical = Chem.MolToSmiles(mol, isomericSmiles=True)
    key = Chem.MolToInchiKey(mol)
    if not key or not re.fullmatch(r"[A-Z]{14}-[A-Z]{10}-[A-Z]", key):
        return None, "invalid_inchikey"
    mass = float(Descriptors.ExactMolWt(mol))
    if not math.isfinite(mass) or mass <= 0:
        return None, "invalid_mass"
    charge = Chem.GetFormalCharge(mol)
    # The SDF mol block is authoritative. Some releases have unparseable SMILES
    # annotations despite a valid mol block and matching reported identity/mass.
    # Keep that annotation for audit and use the original SDF graph as fallback.
    warnings = []
    original = values.get("smiles") or canonical
    original_mol = Chem.MolFromSmiles(original)
    if original_mol is None and values.get("structure_source") == "sdf":
        original = canonical
        warnings.append("source_smiles_parse_failure")
    elif original_mol is None or len(Chem.GetMolFrags(original_mol)) != 1:
        return None, "original_smiles_mismatch"
    else:
        original_key = Chem.MolToInchiKey(original_mol)
        if (original_key[:14] != key[:14] or Chem.GetFormalCharge(original_mol) != charge
                or abs(Descriptors.ExactMolWt(original_mol) - mass) > 1e-5):
            return None, "original_smiles_mismatch"
        if original_key != key:
            warnings.append("source_smiles_fullkey_disagreement")
    reported_key = values.get("inchikey")
    if reported_key:
        valid_key = re.fullmatch(r"[A-Z]{14}(?:-[A-Z]{10}-[A-Z])?", reported_key)
        if not valid_key or reported_key[:14] != key[:14]:
            return None, "identity_mismatch"
    reported_mass = values.get("exact_mass")
    if reported_mass is not None:
        try:
            reported_mass = float(reported_mass)
        except (TypeError, ValueError):
            return None, "invalid_reported_mass"
        if not math.isfinite(reported_mass) or reported_mass <= 0:
            return None, "invalid_reported_mass"
        if abs(reported_mass - mass) > mass_tolerance:
            return None, "mass_mismatch"
    formula = rdMolDescriptors.CalcMolFormula(mol)
    reported_formula = values.get("formula")
    if reported_formula:
        reported_counts = _formula_atoms(reported_formula)
        if reported_counts is not None and reported_counts != _formula_atoms(formula):
            return None, "formula_mismatch"
    if values.get("charge") is not None:
        try:
            expected_charge = float(values["charge"])
        except (TypeError, ValueError):
            return None, "invalid_reported_charge"
        if expected_charge != charge:
            return None, "charge_mismatch"
    return {
        "inchikey14": key[:14], "normalized_smiles": canonical,
        "mass": mass, "exact_mass": mass, "molecular_formula": formula,
        "formal_charge": charge, "original_smiles": original, "raw_inchikey": key,
        "_warnings": warnings,
    }, None


def _json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _union_provenance(left: list, right: list) -> list:
    result, seen = [], set()
    for entry in left + right:
        encoded = _json(entry)
        if encoded not in seen:
            result.append(entry)
            seen.add(encoded)
    return result


def _provenance(row: dict) -> list:
    value = row.get("provenance_json")
    if isinstance(value, str) and value:
        entries = json.loads(value)
        if not isinstance(entries, list) or any(not isinstance(p, dict) for p in entries):
            raise ValueError("provenance_json must contain a list of source records")
        return entries
    # Existing spectral-library catalogs do not necessarily have import metadata.
    return [{"source": _clean(row.get("origin")) or "base", "source_id": _clean(row.get("source_id")), "raw_inchikey": _clean(row.get("raw_inchikey")) or _clean(row.get("inchikey14"))}]


def merge_catalogs(base: pd.DataFrame, extensions: list[pd.DataFrame]) -> pd.DataFrame:
    """Union provenance by raw InChIKey14, keeping the first graph and mass.

    Stereoisomers sharing this scoring key collapse with all source graphs kept
    in provenance. Tautomer canonicalization is deliberately not applied here;
    use the competition's exact normalization separately for submission scoring.
    Every existing base column/value is retained except the added provenance.
    If the union contains formal_charge, missing legacy charges are computed
    from each retained SMILES graph. No structure or mass is changed.
    """
    columns = list(base.columns)
    if "inchikey14" not in columns:
        raise ValueError("base catalog requires inchikey14")
    for extra in extensions:
        if "inchikey14" not in extra.columns:
            raise ValueError("external catalog requires inchikey14")
        columns.extend(c for c in extra.columns if c not in columns)
    if "provenance_json" not in columns:
        columns.append("provenance_json")
    rows, positions = [], {}
    for frame in [base, *extensions]:
        for row in frame.to_dict("records"):
            key = _clean(row.get("inchikey14"))
            if key is None:
                raise ValueError("catalog contains an empty structure key")
            if "formal_charge" in columns:
                charge = row.get("formal_charge")
                if charge is None or pd.isna(charge) or _clean(charge) is None:
                    smiles = _clean(row.get("normalized_smiles"))
                    mol = Chem.MolFromSmiles(smiles) if smiles else None
                    if mol is None or not mol.GetNumAtoms():
                        raise ValueError(f"cannot determine missing formal_charge for {key}: invalid retained normalized_smiles")
                    row["formal_charge"] = Chem.GetFormalCharge(mol)
            provenance = _provenance(row)
            if key in positions:
                existing = rows[positions[key]]
                existing["provenance_json"] = _json(_union_provenance(_provenance(existing), provenance))
            else:
                row["provenance_json"] = _json(provenance)
                positions[key] = len(rows)
                rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def _write_catalog(frame: pd.DataFrame, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError("output must be .parquet or .csv")


def manifest_path(path: str | Path) -> Path:
    return Path(str(path) + ".manifest.json")


def _source_representatives(raw_frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    frame = merge_catalogs(raw_frame, [])
    # Standard InChI connectivity keys can collapse acid/base protonation forms.
    # Prefer a neutral graph only when it genuinely occurs in the source; keep
    # the public base-preserving merge rule unchanged.
    neutral = raw_frame[raw_frame.formal_charge == 0].drop_duplicates("inchikey14").set_index("inchikey14")
    replace = (frame.formal_charge != 0) & frame.inchikey14.isin(neutral.index)
    replacements = int(replace.sum())
    for column in CATALOG_COLUMNS:
        if column not in {"provenance_json", "inchikey14"}:
            frame.loc[replace, column] = frame.loc[replace, "inchikey14"].map(neutral[column]).to_numpy()
    return frame, replacements


def import_catalog(
    path: str | Path, spec: SourceSpec, output_path: str | Path | None = None,
    mass_tolerance: float = 0.01,
) -> tuple[pd.DataFrame, dict]:
    """Validate a local public source; report rejection counts and SHA256 lineage."""
    if not math.isfinite(mass_tolerance) or mass_tolerance < 0:
        raise ValueError("mass_tolerance must be nonnegative and finite")
    path = Path(path)
    kind = _kind(path)
    schema = spec.schema
    if schema is None:
        schema = (SDF_SCHEMAS if kind == "sdf" else TABULAR_SCHEMAS).get(spec.source.lower())
    if schema is None or spec.source.lower() == "coconut" and spec.schema is None:
        raise ValueError("supply an explicit ColumnSchema for this source and release")
    digest = sha256(path)
    if spec.expected_sha256 and digest.lower() != spec.expected_sha256.lower():
        raise ValueError("download SHA256 does not match expected source SHA256")
    archive_members, failures, examples, rows = [], Counter(), [], []
    warning_counts, warning_examples = Counter(), []
    input_rows = 0
    for record, values, mol in _records(path, schema, archive_members):
        input_rows += 1
        try:
            validated, reason = _validate(values, mol, mass_tolerance)
        except (RuntimeError, ValueError, OverflowError):
            validated, reason = None, "structure_processing_error"
        if reason:
            failures[reason] += 1
            if len(examples) < 25:
                examples.append({"record": record, "source_id": values.get("source_id"), "reason": reason})
            continue
        source_id = values.get("source_id") or f"record:{record}"
        warnings = validated.pop("_warnings")
        warning_counts.update(warnings)
        if warnings and len(warning_examples) < 25:
            warning_examples.append({"record": record, "source_id": source_id, "warnings": warnings})
        provenance = {
            "source": spec.source, "source_id": source_id,
            "source_url": spec.source_url, "license": spec.license,
            "release": spec.release, "source_sha256": digest,
            "record": record, "original_smiles": validated["original_smiles"],
            "source_graph_smiles": validated["normalized_smiles"],
            "exact_mass": validated["exact_mass"],
            "formal_charge": validated["formal_charge"],
            "molecular_formula": validated["molecular_formula"],
            "reported_smiles": values.get("smiles"), "warnings": warnings,
            "raw_inchikey": validated["raw_inchikey"],
            "reported_inchikey": values.get("inchikey"),
            "reported_mass": values.get("exact_mass"),
            "reported_formula": values.get("formula"),
            "reported_charge": values.get("charge"),
            "annotations": values.get("annotations", {}),
            "source_metadata": spec.source_metadata,
        }
        validated.update(origin=spec.source, source_id=source_id, provenance_json=_json([provenance]))
        rows.append(validated)
    raw_frame = pd.DataFrame(rows, columns=CATALOG_COLUMNS)
    frame, replacements = _source_representatives(raw_frame)
    manifest = {
        "schema_version": 1, "operation": "public_structure_import",
        "rdkit_version": rdkit.__version__,
        "source": asdict(spec), "resolved_columns": asdict(schema),
        "input_file": str(path.resolve()), "source_sha256": digest,
        "archive_members": archive_members, "input_rows": input_rows,
        "valid_rows": len(rows), "structures": len(frame),
        "duplicate_structure_keys": len(rows) - len(frame),
        "rejected_rows": sum(failures.values()), "rejections": dict(failures),
        "rejection_examples": examples, "mass_tolerance_da": mass_tolerance,
        "warnings": dict(warning_counts), "warning_examples": warning_examples,
        "charged_structures": int((frame.formal_charge != 0).sum()),
        "neutral_representative_replacements": replacements,
        "policy": {
            "graph": "preserve charges, isotopes and tautomers; no salt stripping or neutralization",
            "disconnected": "reject all disconnected structures; no parent-molecule guessing",
            "mass": "recomputed RDKit ExactMolWt for the preserved molecular graph",
            "identity": "raw InChIKey14; prefer first actual neutral source graph when present, else retain first graph; union all source provenance",
            "selection": "public structure consistency only; no test labels or online queries",
            "charged_candidates": "require adduct/charge compatibility during candidate retrieval",
        },
    }
    if output_path is not None:
        output_path = Path(output_path)
        _write_catalog(frame, output_path)
        manifest.update(output_file=str(output_path.resolve()), derived_sha256=sha256(output_path))
        manifest_path(output_path).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return frame, manifest


def _read_catalog(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"source_id": str, "inchikey14": str})
    raise ValueError("catalog inputs must be .parquet or .csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="validate a downloaded public source")
    importer.add_argument("input", type=Path)
    importer.add_argument("--source", required=True)
    importer.add_argument("--source-url", required=True)
    importer.add_argument("--license", required=True)
    importer.add_argument("--release")
    importer.add_argument("--schema", type=Path, help="JSON object of ColumnSchema field -> exact input header")
    importer.add_argument("--source-metadata", type=Path, help="JSON source/license/release evidence to preserve")
    importer.add_argument("--expected-sha256")
    importer.add_argument("--mass-tolerance", type=float, default=0.01)
    importer.add_argument("--output", type=Path, required=True)
    merger = commands.add_parser("merge", help="preserve base structures and union public provenance")
    merger.add_argument("catalogs", nargs="+", type=Path, help="first catalog is the preserved base")
    merger.add_argument("--output", required=True, type=Path)
    merger.add_argument("--prefer-source-neutral", action="store_true", help="external-source-only union: prefer an existing neutral graph; requires public import manifests")
    args = parser.parse_args()
    if args.command == "import":
        schema = ColumnSchema(**json.loads(args.schema.read_text())) if args.schema else None
        metadata = json.loads(args.source_metadata.read_text()) if args.source_metadata else {}
        spec = SourceSpec(args.source, args.source_url, args.license, schema, args.release, args.expected_sha256, metadata)
        _, manifest = import_catalog(args.input, spec, args.output, args.mass_tolerance)
    else:
        frames = [_read_catalog(p) for p in args.catalogs]
        replacements = 0
        if args.prefer_source_neutral:
            for path in args.catalogs:
                sidecar = manifest_path(path)
                if not sidecar.exists() or json.loads(sidecar.read_text()).get("operation") != "public_structure_import":
                    raise ValueError("neutral preference is only supported for manifested public-source imports, never a deployed base")
            frame, replacements = _source_representatives(pd.concat(frames, ignore_index=True))
        else:
            frame = merge_catalogs(frames[0], frames[1:])
        _write_catalog(frame, args.output)
        manifest = {
            "schema_version": 1, "operation": "public_structure_merge",
            "inputs": [{"path": str(p.resolve()), "sha256": sha256(p), "manifest_sha256": sha256(manifest_path(p)) if manifest_path(p).exists() else None} for p in args.catalogs],
            "input_rows": sum(len(f) for f in frames), "structures": len(frame),
            "policy": "prefer first existing neutral public source graph, else preserve first graph; union source provenance by raw InChIKey14" if args.prefer_source_neutral else "preserve first catalog graph and mass; union source provenance by raw InChIKey14",
            "neutral_representative_replacements": replacements,
            "charged_structures": int((frame.formal_charge != 0).sum()) if "formal_charge" in frame else None,
            "output_file": str(args.output.resolve()), "derived_sha256": sha256(args.output),
        }
        manifest_path(args.output).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: manifest[k] for k in ("structures", "derived_sha256")}))


if __name__ == "__main__":
    main()
