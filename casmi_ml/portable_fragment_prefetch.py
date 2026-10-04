"""Two independent JVMs prefetch the same request dictionaries within one deadline."""

import hashlib
import pickle
import tempfile
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from casmi_ml.chembl_routed_inference import extend as serial_extend
from casmi_ml.metfrag_persistent import PersistentMonomerMetFrag


class FragmentPrefetch:
    def __init__(self, deadline):
        self.deadline = deadline
        self.requests = {}
        self.futures = {}
        self.workers = []
        self.local = threading.local()
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.parameters = None

    def record(self, engine, cache_path, candidates, peaks, mass, adduct):
        self.parameters = {
            "jar": engine.jar,
            "cache": engine.cache,
            "classes": engine.runner.command[3].split(":")[0],
            "java": engine.java,
            "threads": engine.threads,
            "timeout": engine.timeout,
            "depth": engine.depth,
        }
        self.requests.setdefault(
            str(cache_path), (cache_path, dict(candidates), list(peaks), mass, adduct)
        )
        return {"status": "complete", "scores": {}}

    def perform(self, request):
        remaining = (
            self.deadline - time.monotonic() if self.deadline is not None else 60
        )
        if remaining <= 0:
            return {"status": "timeout", "scores": {}}
        if not hasattr(self.local, "engine"):
            engine = PersistentMonomerMetFrag(**self.parameters)
            self.local.engine = engine
            self.workers.append(engine)
        engine = self.local.engine
        engine.timeout = min(self.parameters["timeout"], remaining)
        result = engine._run(*request)
        if result.get("status") != "complete":
            raise ValueError(
                "Prefetched physical request failed: " + result.get("status", "unknown")
            )
        return result

    def refill(self):
        while self.waiting and sum(not f.done() for f in self.futures.values()) < 2:
            key = self.waiting.popleft()
            if key not in self.futures:
                self.futures[key] = self.executor.submit(
                    self.perform, self.requests[key]
                )

    def start(self):
        # At most two outstanding requests; later strong-slot original requests
        # must not sit behind the full proposal cohort in an executor queue.
        self.waiting = deque(self.requests)
        self.refill()

    def fetch(self, cache_path, candidates, peaks, mass, adduct):
        key = str(cache_path)
        if key not in self.futures:
            request = (cache_path, dict(candidates), list(peaks), mass, adduct)
            self.futures[key] = self.executor.submit(self.perform, request)
        result = self.futures[key].result()
        self.refill()
        return result

    def close(self):
        self.executor.shutdown(wait=True, cancel_futures=True)
        for engine in self.workers:
            engine.close()


def extend_prefetched(*args, **kwargs):
    from casmi_ml import chembl_routed_inference as base

    config = args[6] if len(args) > 6 else kwargs["config"]
    if config.get("query_residual") or config.get("extended_candidate_compatibility"):
        raise ValueError("Prefetch supports the frozen original critic only")
    if config.get("high_fragment", {}).get("backend") != "persistent":
        raise ValueError("Persistent original physical engine required")
    output = Path(args[4] if len(args) > 4 else kwargs["output"])
    cache = kwargs.get("cache") or str(output) + ".external_fragment_cache"
    pool = FragmentPrefetch(kwargs.get("fragment_deadline"))
    probabilities, scores = {}, {}
    probability_function, score_function = base.group_probability, base.score_group

    def probability(model, group, prep):
        key = hashlib.sha256(pickle.dumps((group, prep), protocol=5)).digest()
        if key not in probabilities:
            probabilities[key] = probability_function(model, group, prep)
        return probabilities[key]

    def score(model, encoder, group, prep, candidates, fps):
        key = hashlib.sha256(
            pickle.dumps((group, prep, candidates, fps), protocol=5)
        ).digest()
        if key not in scores:
            scores[key] = score_function(model, encoder, group, prep, candidates, fps)
        return scores[key]

    class Recorder(PersistentMonomerMetFrag):
        def _run(self, *request):
            return pool.record(self, *request)

    class Consumer(PersistentMonomerMetFrag):
        def _run(self, *request):
            return pool.fetch(*request)

    try:
        with (
            patch.object(base, "group_probability", new=probability),
            patch.object(base, "score_group", new=score),
        ):
            with tempfile.TemporaryDirectory(
                prefix="casmi-prefetch-plan-"
            ) as directory:
                planning_args, planning_kwargs = list(args), dict(kwargs)
                if len(planning_args) > 4:
                    planning_args[4] = Path(directory) / "planning.csv"
                else:
                    planning_kwargs["output"] = Path(directory) / "planning.csv"
                planning_kwargs.update(cache=cache, output_full_rankings=None)
                with patch(
                    "casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=Recorder
                ):
                    serial_extend(*planning_args, **planning_kwargs)
            if pool.requests:
                pool.start()
                with patch(
                    "casmi_ml.metfrag_persistent.PersistentMonomerMetFrag", new=Consumer
                ):
                    return serial_extend(*args, **kwargs)
            return serial_extend(*args, **kwargs)
    finally:
        pool.close()
