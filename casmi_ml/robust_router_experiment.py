"""Count-free routing with an explicit high-similarity retrieval guard."""

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline

from casmi_ml import router_experiment as base
from casmi_ml.ablation import sha
from casmi_ml.data import write_json
from casmi_ml.router_features import FEATURES
from casmi_ml.scale_experiment import sample_keys
from casmi_ml.training import configure

ROOT = Path("artifacts/robust_router_20260929")
PREVIOUS = Path("artifacts/router_20260929")
KEEP = [i for i, name in enumerate(FEATURES) if name != "query_spectra"]


def actions(rows, config, probabilities=None):
    if config["type"] == "baseline":
        return [r["baseline_action"] for r in rows]
    return [
        "H" if r["features"][0] >= config["guard"] or p >= config["threshold"] else "F"
        for r, p in zip(rows, probabilities)
    ]


def classifier(name):
    return make_pipeline(
        ColumnTransformer([("observable", "passthrough", KEEP)], remainder="drop"),
        base.classifier(name),
    )


def protocol():
    ROOT.mkdir(parents=True, exist_ok=True)
    configs = [{"name": "baseline", "type": "baseline"}]
    for learner in ["logistic", "boosted"]:
        for threshold in [0.2, 0.4, 0.6, 0.8]:
            for guard in [0.95, 0.99]:
                configs.append(
                    {
                        "name": f"{learner}_t{threshold}_g{guard}",
                        "type": learner,
                        "high": "H",
                        "low": "F",
                        "threshold": threshold,
                        "guard": guard,
                    }
                )
    spec = {
        "encoder_sha256": sha(base.MODEL),
        "features": [FEATURES[i] for i in KEEP],
        "configs": configs,
        "folds": "SHA256(key) mod 5, same molecule grouped across modes",
        "selection": "maximize unknown OOF MRR; known MRR >= baseline-.001 and top1 >= baseline-.005; prefer baseline on ties",
        "acceptance": "fresh unknown paired CI95 lower >0; known MRR difference >=-.001 and top1 difference >=-.005",
        "fresh_molecules": 2000,
        "motivation": "Remove direct query-count shortcut; explicitly preserve high-similarity reference ranking.",
        "limitations": "Other features can still correlate with query count. Encoder selected on development; only router is OOF. Final new holdout required.",
    }
    path = ROOT / "protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == spec
    write_json(path, spec)
    return spec


def develop(spec):
    path = ROOT / "selection.json"
    if path.exists():
        return json.loads(path.read_text())
    rows = [
        row
        for mode in ["unknown", "known"]
        for row in json.loads((PREVIOUS / f"dev_{mode}_rows.json").read_text())
    ]
    x = np.array([r["features"] for r in rows])
    y = np.array([r["label"] for r in rows])
    folds = np.array(
        [
            int.from_bytes(hashlib.sha256(r["key"].encode()).digest()[:8], "big") % 5
            for r in rows
        ]
    )
    pred = {}
    for name in ["logistic", "boosted"]:
        out = np.empty(len(rows))
        for fold in range(5):
            train = folds != fold
            assert not {rows[i]["key"] for i in np.flatnonzero(train)} & {
                rows[i]["key"] for i in np.flatnonzero(~train)
            }
            model = classifier(name).fit(x[train], y[train])
            out[~train] = model.predict_proba(x[~train])[:, 1]
        pred[name] = out
        np.save(ROOT / f"oof_{name}.npy", out)
    baseline = {
        m: base.score(rows, actions(rows, {"type": "baseline"}), m)[0]
        for m in ["unknown", "known"]
    }
    comparisons = []
    for config in spec["configs"]:
        chosen = actions(rows, config, pred.get(config["type"]))
        r = {m: base.score(rows, chosen, m)[0] for m in ["unknown", "known"]}
        eligible = (
            r["unknown"]["mrr25"] >= baseline["unknown"]["mrr25"]
            and r["known"]["mrr25"] >= baseline["known"]["mrr25"] - 0.001
            and r["known"]["top1"] >= baseline["known"]["top1"] - 0.005
        )
        comparisons.append({"config": config, **r, "eligible": eligible})
    winner = max(
        (r for r in comparisons if r["eligible"]),
        key=lambda r: (r["unknown"]["mrr25"], r["config"]["type"] == "baseline"),
    )
    model_path = None
    if winner["config"]["type"] != "baseline":
        model_path = ROOT / "router.joblib"
        joblib.dump(classifier(winner["config"]["type"]).fit(x, y), model_path)
    selection = {
        "winner": winner,
        "baseline": baseline,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": sha(model_path) if model_path else None,
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "frozen_before_fresh_holdout": True,
    }
    write_json(ROOT / "dev_comparisons.json", comparisons)
    write_json(path, selection)
    pd.DataFrame(
        {
            "key": [r["key"] for r in rows],
            "mode": [r["mode"] for r in rows],
            "fold": folds,
            **pred,
        }
    ).to_csv(ROOT / "oof_predictions.csv", index=False)
    print("FROZEN", json.dumps(selection), flush=True)
    return selection


def fresh_data():
    if (ROOT / "fresh.parquet").exists():
        return
    used = set()
    paths = list(Path("artifacts/experiment").glob("*.parquet"))
    paths += [
        Path("artifacts") / d / f"{s}.parquet"
        for d, names in [
            ("ablation_20260928", ["fresh"]),
            ("scale_20260929", ["fresh", "train60k"]),
            ("router_20260929", ["fresh"]),
        ]
        for s in names
    ]
    # Catalog contains all molecules; exclude only actual prior training/query cohorts.
    paths = [p for p in paths if p.name != "catalog.parquet"]
    for path in paths:
        used.update(pd.read_parquet(path, columns=["inchikey14"]).inchikey14)
    catalog = pd.read_parquet(base.OLD / "catalog.parquet")
    keys = set(
        catalog.loc[
            (catalog.split == "holdout") & ~catalog.inchikey14.isin(used), "inchikey14"
        ].head(2000)
    )
    assert len(keys) == 2000 and not keys & used
    frame = sample_keys(keys, 20261002)
    frame.to_parquet(ROOT / "fresh.parquet", index=False)
    write_json(
        ROOT / "fresh_manifest.json",
        {
            "molecules": len(keys),
            "spectra": len(frame),
            "keys": sorted(keys),
            "excluded_paths": [str(p) for p in paths],
            "prior_overlap": 0,
            "selection_sha256": sha(ROOT / "selection.json"),
            "sha256": sha(ROOT / "fresh.parquet"),
        },
    )


def main():
    configure(42, 4)
    spec = protocol()
    selection = develop(spec)
    base.ROOT = ROOT
    base.fresh_data = fresh_data
    base.chosen_actions = actions
    report = base.evaluate(selection)
    if "known" in report:
        report["accepted"] = (
            report["accepted"]
            and report["known"]["top1"] >= report["known"]["baseline"]["top1"] - 0.005
        )
        write_json(ROOT / "fresh_report.json", report)
    print("FINAL", json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
