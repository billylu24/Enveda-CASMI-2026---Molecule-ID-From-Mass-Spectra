"""Frozen181 individual-slot promotion from label-free reference and fragment evidence."""

from casmi_ml.chembl_high_reference_prefix import reference_supported_prefix
from casmi_ml.generation_slots import insert_generated
from casmi_ml.mass_candidates import mass_centers


def reference_supported_keys(reference, query):
    scores = {}
    for center in mass_centers(query, "charge_aware_union"):
        for key, score in reference.rank(query, center):
            scores[key] = max(scores.get(key, 0), score)
    return {key for key, score in scores.items() if score >= 0.5}


def select_strong_slots(
    original,
    current163,
    proposed,
    critic,
    fragments,
    original_fragments,
    original_fallback,
    observed,
):
    if original_fallback or any(k not in original_fragments for k in original[:3]):
        return current163, [], "strong_original_evidence_fallback"
    maximum = max(original_fragments[k] for k in original[:3])
    if fragments.get(proposed[0], 0) <= maximum:
        return current163, [], "strong_original_gate_fallback"
    supported = [
        k
        for k in proposed[:3]
        if fragments.get(k, 0) > maximum and critic[k] > critic[original[0]] + 0.05
    ]
    result = insert_generated(
        original, supported, reference_supported_prefix(original, observed), 3
    )
    return result, supported, "high_strong_slot_inserted"
