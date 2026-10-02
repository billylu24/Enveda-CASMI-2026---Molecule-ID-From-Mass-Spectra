"""Training-only conditional chemical compatibility, evaluated on novel generation."""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from baseline import formula_mass
from casmi_ml.chemistry import RULES, candidate_support, extract_evidence
from casmi_ml.data import write_json
from casmi_ml.generated_chemical_evidence import COLUMNS
from casmi_ml.generated_score_combination import (
    GENERATED,
    SCORES,
    SOURCE,
    combined_order,
)
from casmi_ml.generated_second_reference import select_prefix
from casmi_ml.generation_slots import insert_generated
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.reference_guard import protects_reference
from casmi_ml.research_budget import StageBudget
from casmi_ml.research_protocol import ROOT, TRAIN, freeze

NAMES = [r.name for r in RULES]


def observed_strength(records):
    strength = np.zeros(len(NAMES))
    index = {name: i for i, name in enumerate(NAMES)}
    for record in records:
        for match in extract_evidence(record)["matches"]:
            i = index[match["rule"]]
            strength[i] = max(strength[i], match["strength"])
    return strength


def structural_support(smiles):
    support = candidate_support(smiles)
    if support is None:
        raise ValueError("Training/candidate structure must be valid")
    return np.array([support[name] for name in NAMES], dtype=float)


def fit_weights(differences, penalty=0.01):
    differences = np.asarray(differences, dtype=float)
    if (
        differences.ndim != 2
        or differences.shape[1] != len(NAMES)
        or not np.isfinite(differences).all()
    ):
        raise ValueError("Invalid chemical training differences")
    if not len(differences):
        raise ValueError("No training hard-negative pairs")

    def objective(weights):
        margin = differences @ weights
        loss = np.logaddexp(0, -margin).mean() + penalty * weights @ weights / 2
        from scipy.special import expit

        gradient = (
            -differences.T @ expit(-margin) / len(differences) + penalty * weights
        )
        return loss, gradient

    result = minimize(objective, np.zeros(len(NAMES)), jac=True, method="L-BFGS-B")
    if not result.success or not np.isfinite(result.x).all():
        raise ValueError("Chemical critic optimizer failed")
    return result.x, float(result.fun)


def train(output, deadline):
    output = Path(output)
    frame = pd.read_parquet(
        TRAIN, columns=COLUMNS + ["normalized_smiles", "molecular_formula"]
    )
    groups = frame.groupby("inchikey14", sort=True).indices
    development = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    if set(groups) & development or len(groups) != 60000:
        raise ValueError("Training/development isolation or60000 scope failed")
    keys = list(groups)
    selected = frame.iloc[[groups[key][0] for key in keys]]
    masses = selected.molecular_formula.map(formula_mass).to_numpy()
    support = np.stack([structural_support(s) for s in selected.normalized_smiles])
    order = np.argsort(masses, kind="stable")
    locations = np.empty(len(order), dtype=int)
    locations[order] = np.arange(len(order))
    strengths = []
    for i, key in enumerate(keys):
        if time.monotonic() >= deadline:
            raise TimeoutError("Chemical critic training preparation budget exhausted")
        records = frame.iloc[groups[key]][COLUMNS[1:]].to_dict("records")
        strengths.append(observed_strength(records))
        if (i + 1) % 5000 == 0:
            print("chemical_training", i + 1, "/", len(keys), flush=True)
    strengths = np.stack(strengths)
    differences, in_window = [], 0
    for i in range(len(keys)):
        pos = locations[i]
        neighborhood = order[max(0, pos - 64) : min(len(keys), pos + 65)]
        neighbors = sorted(
            (int(j) for j in neighborhood if j != i),
            key=lambda j: (abs(masses[j] - masses[i]), keys[j]),
        )[:15]
        tolerance = max(0.006, masses[i] * 35e-6)
        for j in neighbors:
            differences.append(strengths[i] * (support[i] - support[j]))
            in_window += int(abs(masses[i] - masses[j]) <= tolerance)
    weights, loss = fit_weights(differences)
    model = {
        "rule_names": NAMES,
        "weights": weights.tolist(),
        "penalty": 0.01,
        "training_sha256": digest(TRAIN),
        "training_molecules": len(keys),
        "training_pairs": len(differences),
        "pairs_within_mass_window": in_window,
        "training_objective": loss,
        "training_development_key_overlap": 0,
        "training_queries_with_evidence": int((strengths.sum(1) > 0).sum()),
        "architecture": "linear conditional compatibility on observed strength times SMARTS support",
        "adversarial_generator_training": False,
    }
    write_json(output / "model.json", model)
    return model


def run(output, incumbent):
    output, incumbent = Path(output), Path(incumbent)
    output.mkdir(parents=True, exist_ok=True)
    variants = {
        "baseline": 0.0,
        "learned_0.25": 0.25,
        "learned_0.5": 0.5,
        "learned_1": 1.0,
    }
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source_sha256": digest(Path(__file__)),
            "chemistry_sha256": digest(Path("casmi_ml/chemistry.py")),
            "generated_sha256": digest(GENERATED),
            "critic_scores_sha256": digest(SCORES),
            "train_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "variants": variants,
            "training": "60000 training molecules only;15 nearest-mass training negatives; fixedL2.01; no dev fitting",
            "candidate_scope": "novel generation only;0047 routing/adaptive prefix and5 slots frozen",
            "rule_names": NAMES,
            "holdout_used": False,
            "cohort": "repeated_development",
            "truth_used_only_in_metrics": True,
            "new_gpu_training": False,
        },
    )
    started = time.monotonic()
    budget = StageBudget(
        output,
        "chemical_cpu",
        "fit",
        3600,
        limit=3600,
        lock_path=output / "chemical.lock",
    )
    try:
        model = (
            json.loads((output / "model.json").read_text())
            if (output / "model.json").exists()
            else train(output, budget.started + budget.allowance)
        )
    finally:
        budget.close()
    weights = np.array(model["weights"])
    generated = json.loads(GENERATED.read_text())
    candidates = {r["key"]: r["candidates"] for r in generated}
    if len(candidates) != 2000 or len(generated) != 2000:
        raise ValueError("Require2000 complete generated records")
    critic = json.loads(SCORES.read_text())
    ordered = {
        k: combined_order(v, critic.get(k, {}).get("fingerprint", {}), (1, 0, 0.5))
        for k, v in candidates.items()
    }
    frame = pd.read_parquet(ROOT / "researchdev.parquet", columns=COLUMNS)
    groups = frame.groupby("inchikey14", sort=True).indices
    scores = {}
    for key, values in candidates.items():
        strength = observed_strength(
            frame.iloc[groups[key]][COLUMNS[1:]].to_dict("records")
        )
        scores[key] = {
            c["key"]: float((strength * structural_support(c["smiles"])) @ weights)
            for c in values
        }
    write_json(output / "scores.json", scores)
    from casmi_ml.chemistry import rerank

    report = {}
    for mode in ("unknown", "known"):
        rows = json.loads((SOURCE / f"{mode}_records.json").read_text())
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (ROOT / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        observed = set(
            pd.read_parquet(
                Path("artifacts/research_loop/rounds/0001_mass_v2/reference")
                / mode
                / "rows.parquet",
                columns=["inchikey14"],
            ).inchikey14
        )
        report[mode] = {}
        for name, weight in variants.items():
            ranks, pools = {}, {}
            for row in rows:
                if mode == "known" and not row["known"]:
                    continue
                key, base = row["key"], row["variants"]["baseline"]
                reference = protects_reference(
                    base["ranking"], observed, confidence[key]
                )
                selected = base if reference else row["variants"]["expansion_1"]
                prefix = (
                    select_prefix(
                        base["ranking"],
                        observed,
                        confidence[key],
                        "second_unreferenced",
                    )
                    if reference
                    else 2
                )
                novel = [k for k in ordered[key] if k not in set(selected["ranking"])]
                ranked = rerank(
                    novel,
                    {},
                    [],
                    weight,
                    top_n=max(1, len(novel)),
                    fragment_scores=scores[key],
                )
                ranks[key] = insert_generated(selected["ranking"], ranked, prefix, 5)
                pools[key] = selected["pool"] + ordered[key]
            result, per = metrics(ranks, pools)
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    report["diagnostics"] = {
        **model,
        "elapsed_seconds": time.monotonic() - started,
        "candidate_queries_with_different_scores": sum(
            len(set(v.values())) > 1 for v in scores.values()
        ),
        "independent_acceptance": False,
    }
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--incumbent", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.incumbent), indent=2))


if __name__ == "__main__":
    main()
