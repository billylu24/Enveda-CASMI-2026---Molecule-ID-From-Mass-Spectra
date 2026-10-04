"""Overlap same original physical requests with discovery, without changing scores."""

from collections import deque
from unittest.mock import patch

from casmi_ml import portable_fragment_prefetch as original


class StreamingFragmentPrefetch(original.FragmentPrefetch):
    def __init__(self, deadline):
        super().__init__(deadline)
        self.waiting = deque()

    def record(self, engine, cache_path, candidates, peaks, mass, adduct):
        key = str(cache_path)
        fresh = key not in self.requests
        result = super().record(engine, cache_path, candidates, peaks, mass, adduct)
        if fresh:
            self.waiting.append(key)
        self.refill()
        return result

    def start(self):
        # Discovery already started bounded work; retain each request's original
        # deadline, inputs and result and only fill newly available capacity.
        self.refill()


def extend_streamed(*args, **kwargs):
    with patch.object(original, "FragmentPrefetch", new=StreamingFragmentPrefetch):
        return original.extend_prefetched(*args, **kwargs)
