"""Mean normalized evidence across informative spectra, without cross-spectrum maxima."""

import math
import time

from casmi_ml.metfrag import record_status


def normalized_scores(scores, candidates):
    values = {k: float(scores.get(k, 0.0)) for k in candidates}
    if not values or any(not math.isfinite(v) or v < 0 for v in values.values()):
        return None
    maximum = max(values.values())
    if maximum <= 0 or len(set(values.values())) < 2:
        return None
    return {k: v / maximum for k, v in values.items()}


def score_group_mean(fragmenter, records, candidates, deadline=None):
    """Equal weight per usable spectrum; missing or tied spectra give no evidence."""
    total = {k: 0.0 for k in candidates}
    count = 0
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
            if deadline is not None and time.monotonic() >= deadline:
                return {}, True
            if result["status"] != "complete":
                continue
            evidence = normalized_scores(result["scores"], candidates)
            if evidence is not None:
                count += 1
                for key, value in evidence.items():
                    total[key] += value
        return (
            {k: value / count for k, value in total.items()} if count else {}
        ), False
    finally:
        fragmenter.timeout = original_timeout
