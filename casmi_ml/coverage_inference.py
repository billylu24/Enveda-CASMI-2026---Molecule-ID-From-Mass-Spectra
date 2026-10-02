"""Shared expanded chemical ranking used by controlled experiments and inference."""

from casmi_ml.chemistry_experiment import chemical_ranking
from casmi_ml.mass_candidates import candidate_window, hypothesis_baseline
from casmi_ml.metfrag import score_group
from casmi_ml.ranking import neural_rank, rrf
from casmi_ml.secondary_inference import low_confidence_rank


def expanded_chemical_rank(
    group,
    historical,
    probability,
    index,
    reference,
    structures,
    fragmenter,
    deadline=None,
):
    pool, fps = candidate_window(index, group, "charge_aware_union")
    current = hypothesis_baseline(group, pool, fps, reference, "charge_aware_union")
    neural = neural_rank(probability, pool.inchikey14.tolist(), fps)
    raw = low_confidence_rank(historical, current, neural, 0.75)
    scores, fallback = score_group(
        fragmenter,
        group.to_dict("records"),
        {k: structures[k] for k in raw[:100]},
        deadline,
    )
    ranking = chemical_ranking(
        {"base": raw, "confidence": 0.0, "fragment_scores": scores}, "fragment", 0.5
    )
    return ranking, pool, fallback


def merge_expanded(original, expanded, weight, fallback=False):
    if not 0 <= weight <= 1:
        raise ValueError("Expansion fusion weight must be in [0,1]")
    return original if fallback else rrf([original, expanded], [1 - weight, weight])
