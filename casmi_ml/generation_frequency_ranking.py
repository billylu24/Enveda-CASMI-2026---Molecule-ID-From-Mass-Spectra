"""Tie-aware structure-frequency rank fusion shared by research and deployment."""

from casmi_ml.chemistry import rerank


def ranked_candidates(candidates, weight):
    original = [c["key"] for c in candidates]
    if any(c.get("sample_count", 0) < 1 for c in candidates):
        raise ValueError("Require measured sampling frequency, not inferred counts")
    return rerank(
        original,
        {},
        [],
        weight,
        top_n=max(1, len(original)),
        fragment_scores={c["key"]: c["sample_count"] for c in candidates},
    )
