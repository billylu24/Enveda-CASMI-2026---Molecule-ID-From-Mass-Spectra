"""Review-only bounded physical comparison; publisher policy remains unchanged."""

import math
import struct


def positive_ulp_distance(left, right):
    if any(
        type(v) not in (int, float) or not math.isfinite(v) or v < 0
        for v in (left, right)
    ):
        raise ValueError("Finite nonnegative numeric scores required")
    if left == right:
        return 0
    bits = [struct.unpack(">Q", struct.pack(">d", float(v)))[0] for v in (left, right)]
    return abs(bits[0] - bits[1])


def compare_dictionaries(local, platform, *, max_ulp=0, max_relative=0.0):
    """Require265 complete identical input IDs/metadata/keys/orders, bound every score.

    This does not authorize publication. Default preserves exact comparison;
    a caller must explicitly supply BOTH bounds for numerical equivalence review.
    """
    if (
        type(max_ulp) is not int
        or not 0 <= max_ulp <= 2
        or not math.isfinite(max_relative)
        or not 0 <= max_relative <= 1e-15
    ):
        raise ValueError("At most2ULP and1e-15 relative bounded review supported")
    if len(local) != 265 or len(platform) != 265 or local.keys() != platform.keys():
        raise ValueError("All265 matching physical request IDs required")
    differences = []
    scores = 0
    for name, left in local.items():
        right = platform[name]
        if left.get("status") != "complete" or right.get("status") != "complete":
            raise ValueError("Every physical request must be complete")
        if {k: v for k, v in left.items() if k != "scores"} != {
            k: v for k, v in right.items() if k != "scores"
        }:
            raise ValueError("Physical request non-score fields differ")
        a, b = left.get("scores"), right.get("scores")
        if not isinstance(a, dict) or not isinstance(b, dict) or a.keys() != b.keys():
            raise ValueError("Physical candidate keysets differ")
        for key, x in a.items():
            y = b[key]
            ulp = positive_ulp_distance(x, y)
            scores += 1
            if x != y:
                relative = abs(x - y) / max(abs(x), abs(y))
                if ulp > max_ulp or relative > max_relative:
                    raise ValueError(
                        "Physical scores exceed explicit ULP or relative bound"
                    )
                differences.append((ulp, relative, abs(x - y)))
        if sorted(a, key=lambda k: (-a[k], k)) != sorted(b, key=lambda k: (-b[k], k)):
            raise ValueError("Physical candidate score order changed")
    return {
        "valid": True,
        "requests": 265,
        "candidate_scores": scores,
        "complete_all_requests": True,
        "same_non_score_fields_and_keysets": True,
        "all_candidate_orders_exact": True,
        "all_raw_dictionaries_exact": not differences,
        "different_values": len(differences),
        "max_ulp_distance": max((r[0] for r in differences), default=0),
        "max_relative_difference": max((r[1] for r in differences), default=0.0),
        "max_absolute_difference": max((r[2] for r in differences), default=0.0),
        "explicit_max_ulp": max_ulp,
        "explicit_max_relative": max_relative,
        "publisher_authorization": False,
        "scope": "Review-only comparison, not publication authorization. Publisher strict default remains unchanged pending user choice.",
    }
