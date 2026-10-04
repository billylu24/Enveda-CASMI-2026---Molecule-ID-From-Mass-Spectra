"""Plan unchanged physical requests as original generated molecule handoffs complete."""

import hashlib
import inspect
import json
import pickle
import queue
import subprocess
import tempfile
import threading
import time
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import torch

from casmi_ml import chembl_routed_inference as base
from casmi_ml import parallel_generation_runtime as generation
from casmi_ml import portable_fragment_prefetch as physical
from casmi_ml.data import write_json
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag
from casmi_ml.ranking import ReferenceIndex
from casmi_ml.streaming_fragment_prefetch import StreamingFragmentPrefetch


class CompletedHandoffs:
    """Exactly-once import of atomic local worker events, independent of arrival order."""

    def __init__(self):
        self.lock = threading.Lock()
        self.paths = []
        self.seen = set()
        self.rows = {}
        self.queue = queue.Queue()

    def add(self, path):
        with self.lock:
            self.paths.append(Path(path))

    def collect(self):
        with self.lock:
            for path in self.paths:
                for event in sorted(path.glob("[0-9]*.json")):
                    if event in self.seen:
                        continue
                    row = json.loads(event.read_text())
                    if set(row) != {"molecule_id", "smiles"}:
                        raise ValueError("Original complete handoff event required")
                    key = row["molecule_id"]
                    if key in self.rows:
                        raise ValueError("Duplicate generated molecule event")
                    self.rows[key] = row
                    self.seen.add(event)
                    self.queue.put(row)


class SharedOriginalScoring:
    """Reuse immutable full catalog and original inference values within one invocation."""

    def __init__(self, root, config):
        self.catalog_path = (Path(root) / config["catalog"]).resolve()
        self.catalog = self.index = None
        self.encoder = None
        self.hashes = {}
        self.bound_files = {
            (Path(root) / config[k]).resolve()
            for k in ("catalog", "encoder", "critic", "prior", "jar")
        }
        self.probabilities = {}
        self.scores = {}
        self.original_threads = torch.get_num_threads()

    def enter(self, stack):
        read, constructor, loader = (
            base.pd.read_parquet,
            base.CandidateIndex,
            base.load_deployment_checkpoint,
        )
        original_digest = base.digest
        probability, score, configure = (
            base.group_probability,
            base.score_group,
            base.configure,
        )

        def bound_digest(path):
            resolved = Path(path).resolve()
            if resolved not in self.bound_files:
                return original_digest(path)
            st = resolved.stat()
            signature = (st.st_ino, st.st_size, st.st_mtime_ns, st.st_ctime_ns)
            if resolved not in self.hashes:
                self.hashes[resolved] = (signature, original_digest(path))
            if self.hashes[resolved][0] != signature:
                raise ValueError("Immutable bound inference asset changed")
            return self.hashes[resolved][1]

        def read_catalog(path, *args, **kwargs):
            if (
                not isinstance(path, (str, Path))
                or Path(path).resolve() != self.catalog_path
            ):
                return read(path, *args, **kwargs)
            if args or kwargs:
                raise ValueError("Unchanged whole catalog read required")
            if self.catalog is None:
                self.catalog = read(path)
            return self.catalog

        def index(frame):
            if frame is not self.catalog:
                raise ValueError("Only original whole catalog may reuse index")
            if self.index is None:
                self.index = constructor(frame)
            return self.index

        def encoder(*args, **kwargs):
            if self.encoder is None:
                self.encoder = (args, kwargs, loader(*args, **kwargs))
            if args != self.encoder[0] or kwargs != self.encoder[1]:
                raise ValueError("Original encoder path/arguments changed")
            return self.encoder[2]

        def probabilities(model, group, prep):
            key = hashlib.sha256(pickle.dumps((group, prep), protocol=5)).digest()
            if key not in self.probabilities:
                self.probabilities[key] = probability(model, group, prep)
            return self.probabilities[key]

        def scores(model, encoder, group, prep, candidates, fps):
            key = hashlib.sha256(
                pickle.dumps((group, prep, candidates, fps), protocol=5)
            ).digest()
            if key not in self.scores:
                self.scores[key] = score(model, encoder, group, prep, candidates, fps)
            return self.scores[key]

        for target, name, value in [
            (base, "digest", bound_digest),
            (base.pd, "read_parquet", read_catalog),
            (base, "CandidateIndex", index),
            (base, "load_deployment_checkpoint", encoder),
            (base, "group_probability", probabilities),
            (base, "score_group", scores),
            (
                base,
                "configure",
                lambda seed=42, threads=4: configure(seed=seed, threads=1),
            ),
        ]:
            stack.enter_context(patch.object(target, name, new=value))
        stack.callback(torch.set_num_threads, self.original_threads)


def generate_with_physical_planning(
    args,
    kwargs,
    *,
    root,
    config,
    deadline,
    total_deadline,
    final_output,
    force_partition=False,
):
    if config.get("query_residual") or config.get("extended_candidate_compatibility"):
        raise ValueError("Only unchanged original critic may overlap")
    parameters = inspect.signature(generation.original_predict).bind(*args, **kwargs)
    parameters.apply_defaults()
    p = parameters.arguments
    test = pd.read_parquet(p["test_path"])
    if {"inchikey14", "normalized_smiles", "fingerprint", "molecular_formula"} & set(
        test
    ):
        raise ValueError("Only supplied unlabeled query input allowed")
    routing = pd.read_csv(p["routing_csv"] or str(p["baseline_csv"]) + ".routing.csv")
    reference = ReferenceIndex(str(p["baseline_csv"]) + ".external_reference")
    final_output = Path(final_output)
    cache = str(final_output) + ".external_fragment_cache"
    pool = StreamingFragmentPrefetch(deadline)
    handoffs = CompletedHandoffs()
    stop = threading.Event()
    errors = []
    original_popen = subprocess.Popen
    shared = SharedOriginalScoring(root, config)

    class Recorder(PersistentMonomerMetFrag):
        def _run(self, *request):
            return pool.record(self, *request)

    def plan(directory):
        try:
            while True:
                if stop.is_set() and handoffs.queue.empty():
                    return
                try:
                    first = handoffs.queue.get(timeout=0.02)
                except queue.Empty:
                    handoffs.collect()
                    continue
                batch = [first]
                while len(batch) < 16:
                    try:
                        batch.append(handoffs.queue.get_nowait())
                    except queue.Empty:
                        break
                selected = {row["molecule_id"] for row in batch}
                query = test[test.molecule_id.isin(selected)]
                records = {row["molecule_id"]: row for row in batch}
                order = list(query.molecule_id.unique())
                ordered = [records[key] for key in order]
                query.to_parquet(directory / "test.parquet", index=False)
                pd.DataFrame(
                    [
                        {
                            "molecule_id": key,
                            "smiles": ";".join(records[key]["smiles"][:25]),
                        }
                        for key in order
                    ]
                ).to_csv(directory / "baseline.csv", index=False)
                write_json(directory / "full.json", ordered)
                routing[routing.molecule_id.isin(selected)].to_csv(
                    directory / "routing.csv", index=False
                )
                with patch(
                    "casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=Recorder
                ):
                    physical.serial_extend(
                        directory / "test.parquet",
                        directory / "baseline.csv",
                        directory / "full.json",
                        directory / "routing.csv",
                        directory / "planning.csv",
                        root,
                        config,
                        deadline=total_deadline,
                        fragment_deadline=deadline,
                        cache=cache,
                        reference=reference,
                    )
        except Exception as error:  # noqa: BLE001 - relay unchanged thread failure to caller
            errors.append(error)
            stop.set()

    class Child:
        def __init__(self, process):
            self.process = process

        def __getattr__(self, name):
            return getattr(self.process, name)

        def wait(self, *args, **kwargs):
            result = self.process.wait(*args, **kwargs)
            # Collect all final files BEFORE original parallel_predict deletes its
            # temporary worker directories. Planning uses the in-memory queue.
            handoffs.collect()
            return result

    def popen(command, *args, **kwargs):
        if len(command) >= 5 and command[1:3] == [
            "-m",
            "casmi_ml.parallel_generation_runtime",
        ]:
            command = list(command)
            command[2] = "casmi_ml.generation_handoff_stream"
            handoffs.add(Path(command[4]).parent / "handoff_events")
            return Child(original_popen(command, *args, **kwargs))
        return original_popen(command, *args, **kwargs)

    started = time.monotonic()
    stack = ExitStack()
    planner = None
    try:
        shared.enter(stack)
        with tempfile.TemporaryDirectory(
            prefix="casmi-generation-physical-plan-"
        ) as folder:
            planner = threading.Thread(target=plan, args=(Path(folder),), daemon=True)
            planner.start()
            with patch.object(generation.subprocess, "Popen", new=popen):
                result = generation.parallel_predict(
                    *args, **kwargs, force_partition=force_partition
                )
            stop.set()
            planner.join(timeout=max(0.001, deadline - time.monotonic()))
            if planner.is_alive():
                raise TimeoutError(
                    "Original physical deadline expired during early planning"
                )
            if errors:
                raise errors[0]
            expected = json.loads(Path(p["output_full_rankings"]).read_text())
            if handoffs.rows != {row["molecule_id"]: row for row in expected}:
                raise ValueError(
                    "All streamed completed handoffs must equal original full generation output"
                )
        return (
            result,
            pool,
            stack,
            {
                "emitted_molecules": len(handoffs.rows),
                "planned_requests_at_generation_end": len(pool.requests),
                "generation_and_early_planning_seconds": time.monotonic() - started,
                "original_shared_fragment_deadline": deadline,
            },
        )
    except BaseException:
        stop.set()
        if planner is not None:
            planner.join(timeout=60)
        pool.close()
        stack.close()
        raise
