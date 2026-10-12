"""Metric identity fusion logic implementing official CASMI26 evaluation rules."""

from typing import Callable, Dict, List, Optional, Tuple


def metric_fusion(
    first: List[str],
    second: List[str],
    identity: Callable[[str], Optional[str]],
    alpha: float = 0.6,
    offset: float = 3.0,
    limit: int = 40,
) -> Tuple[List[str], List[float], int]:
    """Combine two candidate lists with structure-level one-vote-per-identity RRF.

    Eliminates tautomer and protonation state duplicate-vote bias in multi-engine fusion.

    Args:
        first: Top candidates from primary engine (e.g. v4g).
        second: Top candidates from secondary engine (e.g. Engine 2).
        identity: Function mapping SMILES to canonical metric identity key.
        alpha: Weight multiplier for the second engine.
        offset: RRF rank smoothing constant (default 3.0).
        limit: Strict cutoff window (default 40).

    Returns:
        Tuple of (selected SMILES, scores, count of collapsed duplicate votes).
    """
    scores: Dict[str, float] = {}
    representatives: Dict[str, str] = {}
    collapsed = 0

    for candidates, weight in ((first, 1.0), (second, alpha)):
        seen = set()
        for candidate in candidates:
            key = identity(candidate)
            if key is None:
                continue
            if key in seen:
                collapsed += 1
                continue
            seen.add(key)
            # RRF score based on unique rank position
            scores[key] = scores.get(key, 0.0) + weight / (offset + len(seen))
            representatives.setdefault(key, candidate)

    ordered = sorted(scores, key=lambda k: -scores[k])[:limit]
    return (
        [representatives[k] for k in ordered],
        [scores[k] for k in ordered],
        collapsed,
    )
