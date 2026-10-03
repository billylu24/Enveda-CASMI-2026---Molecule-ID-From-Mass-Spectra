"""Experimental sequential JVM reuse with bounded per-request timeout and restart."""

import atexit
import csv
import json
import queue
import subprocess
import tempfile
import threading
from collections import deque
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors

from casmi_ml.data import write_json
from casmi_ml.metfrag_monomer import MonomerMetFrag


class PersistentRunner:
    def __init__(self, java, jar, classes, max_requests=100):
        self.command = [
            java,
            "-Xmx1g",
            "-cp",
            str(Path(classes).resolve()) + ":" + str(jar),
            "MetFragWorker",
        ]
        self.process = None
        self.requests = 0
        self.max_requests = max_requests
        atexit.register(self.close)

    def close(self):
        p = self.process
        self.process = None
        if p is not None:
            if p.poll() is None:
                p.kill()
            p.wait()
            for stream in (p.stdin, p.stdout):
                if stream is not None:
                    stream.close()

    def _start(self):
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        p = self.process
        self.events = events = queue.Queue()
        self.tail = tail = deque(maxlen=100)
        self.requests = 0

        def reader():
            for line in p.stdout:
                if line.startswith("CASMI_METFRAG_DONE\t"):
                    events.put(line.rstrip("\r\n").split("\t", 1)[1])
                else:
                    tail.append(line[-500:])
            events.put(None)

        threading.Thread(target=reader, daemon=True).start()

    def run(self, parameters, timeout):
        if "\n" in str(parameters) or "\r" in str(parameters):
            raise ValueError("Parameter paths cannot contain newlines")
        if (
            self.process is None
            or self.process.poll() is not None
            or self.requests >= self.max_requests
        ):
            self.close()
            self._start()
        p = self.process
        try:
            p.stdin.write(str(parameters) + "\n")
            p.stdin.flush()
            event = self.events.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise subprocess.TimeoutExpired(self.command, timeout) from None
        except (BrokenPipeError, OSError):
            event = None
        self.requests += 1
        detail = "".join(self.tail)[-3000:]
        if event != str(parameters):
            self.close()
            return subprocess.CompletedProcess(self.command, 1, detail, "")
        return subprocess.CompletedProcess(self.command, 0, detail, "")


class PersistentMonomerMetFrag(MonomerMetFrag):
    def __init__(self, jar, cache, *, classes, threads=1, **kwargs):
        if type(threads) is not int or threads not in (1, 2):
            raise ValueError("MetFrag candidate threads must be 1 or 2")
        self.threads = threads
        super().__init__(jar, cache, **kwargs)
        self.runner = PersistentRunner(self.java, self.jar, classes)

    def close(self):
        self.runner.close()

    def _run(self, cache_path, candidates, peaks, mass, adduct):
        if self.threads != 1:
            # Keep unvalidated execution variants out of legacy physical caches.
            cache_path = cache_path.with_name(
                f"{cache_path.stem}.threads{self.threads}.json"
            )
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
                "NumberThreads": self.threads,
                "UseSmiles": "True",
            }
            (work / "params.txt").write_text(
                "".join(f"{k} = {v}\n" for k, v in params.items())
            )
            try:
                run = self.runner.run(work / "params.txt", self.timeout)
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
