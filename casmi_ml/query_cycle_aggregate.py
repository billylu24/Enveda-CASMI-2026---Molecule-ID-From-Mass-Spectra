"""Summarize finalized paired query architecture experiments without private rows."""

import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.research_loop import Controller

CYCLES = {
    "single_contextual": "diagnostic_query_attribution.json",
    "strict_mass": "diagnostic_query_strict_mass_attribution.json",
    "same_ion_measured_pair": "query_multispectrum_attribution.json",
    "chemical_only_correction": "query_chemical_only_attribution.json",
    "balanced_modalities": "query_balanced_attribution.json",
    "weak_peak_loss_attenuation": "query_strong_peak_attribution.json",
    "deployed_admission_margin": "query_admission_attribution.json",
    "all_measured_group": "query_all_spectra_attribution.json",
    "candidate_structural_compatibility": "candidate_compatibility_attribution.json",
    "extended_candidate_chemistry": "extended_candidate_attribution.json",
}


def refresh():
    c = Controller()
    if not c.local_only():
        raise ValueError("Query cycle aggregate is local-only")
    state = c.read()
    cycles = {}
    for name, filename in CYCLES.items():
        path = Path("results/research_loop") / filename
        if not path.exists():
            continue
        value = json.loads(path.read_text())
        arms = value.get("arms", {})
        normalized = {}
        for arm, entry in arms.items():
            metrics = entry.get("metrics", {})
            if not metrics:
                continue
            unknown = metrics["unknown"]["mrr25"]
            known = metrics["known"]["mrr25"]
            normalized[arm] = {
                "round": entry.get("round"),
                "unknown_mrr25": unknown,
                "known_mrr25": known,
                "known_top1": metrics["known"]["top1"],
                "unknown_gain_vs_current_best": unknown
                - state["development_best"]["metrics"]["unknown"]["mrr25"],
            }
        if normalized:
            cycles[name] = {
                "arms": normalized,
                "chemical_minus_mask_unknown_mrr25": value.get(
                    "chemical_minus_mask_unknown_mrr25"
                ),
            }
    result = {
        "local_only": True,
        "remote_publication": False,
        "independent_acceptance": False,
        "current_best": state["development_best"],
        "cycles": cycles,
        "scope": "Finalized complete repeated development arms only; training-only diagnostics and incomplete systems excluded, no independent generalization claim",
        "gates": "Unknown absolute+.0005 AND relative+2% vs current best; known MRRdrop<=.001 andTop1drop<=.005, completed0206/historical181 protection, no budgetfallback or newfailed physical contexts",
    }
    write_json("results/research_loop/query_cycle_aggregate.json", result)
    return result


if __name__ == "__main__":
    print(json.dumps(refresh(), indent=2))
