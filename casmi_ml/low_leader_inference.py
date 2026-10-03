"""Label-free promotion under the unchanged low-route evidence gates."""

from casmi_ml.generation_slots import insert_generated

LOW_LEADER_RULE = {"supported_leader_first": True, "original_prefix": 3, "slots": 3}


def select_low_fragment(
    current, proposed, critic, fragments, fallback, leader_first=False
):
    # The caller supplies fragment-sorted proposals. All eligibility checks are
    # observable and are shared by the historical insertion and new promotion.
    from casmi_ml.chembl_fragment_pilot import informative_fragments

    status = "fragment_budget_fallback" if fallback else "fragment_evidence_fallback"
    if (
        not current
        or fallback
        or not informative_fragments(proposed, fragments)
        or fragments.get(proposed[0], 0) <= fragments.get(current[0], 0)
        or critic[proposed[0]] <= critic[current[0]] + 0.05
    ):
        return current, [], status
    inserted = proposed[:3]
    result = insert_generated(current, proposed, 3, 3)
    if leader_first:
        leader = inserted[0]
        result = [leader, *[key for key in result if key != leader]]
        return result, inserted, "low_supported_leader_first"
    return result, inserted, "low_fragment_inserted"
