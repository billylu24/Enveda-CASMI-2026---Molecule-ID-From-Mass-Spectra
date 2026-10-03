"""Validate the original protected prefix, including short original rankings."""


def validate_tail(prior, current, result, confidence):
    if confidence < 0.5:
        if result != current:
            raise ValueError("Frozen low arm changed")
    elif result[: min(10, len(prior))] != prior[:10]:
        raise ValueError("Original protected prefix changed")
