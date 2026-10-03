"""Validate the original protected prefix, including short original rankings."""


def validate_tail(prior, current, result, confidence, prefix=10):
    if confidence < 0.5:
        if result != current:
            raise ValueError("Frozen low arm changed")
    elif result[: min(prefix, len(prior))] != prior[:prefix]:
        raise ValueError("Original protected prefix changed")
