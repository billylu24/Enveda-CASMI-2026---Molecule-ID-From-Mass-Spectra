"""Offline MetFrag adapter with bounded execution and content-addressed results."""

import csv
import fcntl
import hashlib
import json
import subprocess
import tempfile
import time
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi_ml.chemistry import clean_peaks, high_resolution, neutral_mass
from casmi_ml.data import write_json

VERSION = "2.6.11"
URL = f"https://github.com/ipb-halle/MetFragRelaunched/releases/download/v{VERSION}/MetFragCommandLine-{VERSION}.jar"


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for b in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


class MetFrag:
    def __init__(self, jar, cache, timeout=60, java="java", depth=2):
        if type(depth) is not int or depth not in (2, 3):
            raise ValueError("MetFrag depth must be 2 or 3")
        self.depth = depth
        self.jar = Path(jar).resolve()
        if not self.jar.is_file():
            raise FileNotFoundError(self.jar)
        self.sha256 = digest(self.jar)
        self.cache = Path(cache)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout, self.java = timeout, java

    def score(self, row, candidates):
        """Candidates are key -> SMILES; no labels/formula from query are accepted."""
        mass = neutral_mass(row)
        adduct = row.get("adduct")
        if (
            mass is None
            or adduct not in ["[M+H]+", "[M-H]-"]
            or not high_resolution(row.get("instrument_type"))
        ):
            return {"status": "unsupported_ion_or_resolution", "scores": {}}
        mz, intensity = clean_peaks(row)
        valid = mz < float(row["precursor_mz"]) - 0.01
        peaks = list(zip(mz[valid].tolist(), intensity[valid].tolist()))
        if not peaks or not candidates:
            return {"status": "empty", "scores": {}}
        payload = {
            "jar": self.sha256,
            "peaks": peaks,
            "mass": mass,
            "adduct": adduct,
            "candidates": sorted(candidates.items()),
            "ppm": 10,
            "absolute_da": 0.002,
            "depth": self.depth,
        }
        key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
        cache_path = self.cache / f"{key}.json"
        with (self.cache / f"{key}.lock").open("a") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            try:
                return self._run(cache_path, candidates, peaks, mass, adduct)
            finally:
                fcntl.flock(lock, fcntl.LOCK_UN)

    def _run(self, cache_path, candidates, peaks, mass, adduct):
        if cache_path.exists():
            return json.loads(cache_path.read_text())

        with tempfile.TemporaryDirectory(prefix="casmi-metfrag-") as directory:
            work = Path(directory)
            with (work / "candidates.csv").open("w", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(
                    [
                        "Identifier",
                        "InChI",
                        "MonoisotopicMass",
                        "MolecularFormula",
                        "SMILES",
                    ]
                )
                for cid, smiles in candidates.items():
                    mol = Chem.MolFromSmiles(smiles)
                    if mol:
                        writer.writerow(
                            [
                                cid,
                                Chem.MolToInchi(mol),
                                Descriptors.ExactMolWt(mol),
                                Chem.rdMolDescriptors.CalcMolFormula(mol),
                                smiles,
                            ]
                        )
            (work / "peaks.txt").write_text(
                "".join(f"{m:.9f} {i * 100:.6f}\n" for m, i in peaks)
            )
            params = {
                "PeakListPath": str(work / "peaks.txt"),
                "MetFragDatabaseType": "LocalCSV",
                "LocalDatabasePath": str(work / "candidates.csv"),
                "NeutralPrecursorMass": mass,
                "PrecursorIonType": adduct,
                "DatabaseSearchRelativeMassDeviation": 1000000,
                "FragmentPeakMatchAbsoluteMassDeviation": 0.002,
                "FragmentPeakMatchRelativeMassDeviation": 10,
                "MetFragScoreTypes": "FragmenterScore",
                "MetFragScoreWeights": "1.0",
                "MetFragCandidateWriter": "CSV",
                "ResultsPath": str(work),
                "SampleName": "result",
                "MaximumTreeDepth": self.depth,
                "NumberThreads": 1,
                "UseSmiles": "True",
            }
            (work / "params.txt").write_text(
                "".join(f"{k} = {v}\n" for k, v in params.items())
            )
            try:
                run = subprocess.run(
                    [
                        self.java,
                        "-Xmx1g",
                        "-jar",
                        str(self.jar),
                        str(work / "params.txt"),
                    ],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                # Do not cache transient failures as permanent negative evidence.
                return {"status": "timeout", "scores": {}}
            outputs = list(work.glob("result*.csv"))
            if run.returncode or not outputs:
                return {
                    "status": "failed",
                    "scores": {},
                    "detail": (run.stdout + run.stderr)[-3000:],
                }
            scores = {}
            with outputs[0].open() as file:
                for record in csv.DictReader(file):
                    cid = record["Identifier"]
                    scores[cid] = float(
                        record.get("FragmenterScore") or record.get("Score") or 0.0
                    )
            result = {"status": "complete", "scores": scores, "jar_sha256": self.sha256}
            write_json(cache_path, result)
            return result


def record_status(fragmenter, result):
    counts = vars(fragmenter).get("status_counts", {})
    status = result.get("status", "unspecified")
    counts[status] = counts.get(status, 0) + 1
    fragmenter.status_counts = counts


def score_group(fragmenter, records, candidates, deadline=None):
    """Bounded sequential Java scoring; an interrupted group uses retrieval fallback."""
    scores = {}
    original_timeout = fragmenter.timeout
    try:
        for row in records:
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return {}, True
                fragmenter.timeout = min(original_timeout, remaining)
            result = fragmenter.score(row, candidates)
            record_status(fragmenter, result)
            for key, value in result["scores"].items():
                scores[key] = max(scores.get(key, 0.0), value)
            if deadline is not None and time.monotonic() >= deadline:
                return {}, True
    finally:
        fragmenter.timeout = original_timeout
    return scores, False
