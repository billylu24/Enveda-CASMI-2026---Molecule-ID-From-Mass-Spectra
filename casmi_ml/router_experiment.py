"""Molecule-grouped calibration of frozen spectral/neural retrieval actions."""

import argparse
import gc
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from casmi_ml.ablation import sha
from casmi_ml.data import write_json
from casmi_ml.experiment import bootstrap_difference
from casmi_ml.ranking import center_mass
from casmi_ml.router_features import FEATURES, router_features, rr
from casmi_ml.scale_experiment import (
    ABLATION,
    COCONUT,
    OLD,
    Ranker,
    dev_rankers,
    predict_cpu,
    sample_keys,
)
from casmi_ml.secondary_inference import load_deployment_checkpoint, low_confidence_rank
from casmi_ml.training import configure

ROOT = Path("artifacts/router_20260929")
SCALE = Path("artifacts/scale_20260929")
MODEL = SCALE / "runs/wide_enhanced_60000/model.pt"


def policy_configs():
    configs = [{"name": "baseline", "type": "baseline"}]
    for high in ["H", "U"]:
        for low in ["F", "N"]:
            for threshold in [0.35, 0.5, 0.65, 0.8, 0.9]:
                configs.append(
                    {
                        "name": f"fixed_{high}_{low}_{threshold}",
                        "type": "fixed",
                        "high": high,
                        "low": low,
                        "threshold": threshold,
                    }
                )
    for learner in ["logistic", "boosted"]:
        for high in ["H", "U"]:
            for low in ["F", "N"]:
                for threshold in [0.2, 0.4, 0.6, 0.8]:
                    configs.append(
                        {
                            "name": f"{learner}_{high}_{low}_{threshold}",
                            "type": learner,
                            "high": high,
                            "low": low,
                            "threshold": threshold,
                        }
                    )
    return configs


def protocol():
    ROOT.mkdir(parents=True, exist_ok=True)
    spec = {
        "version": 1,
        "encoder": str(MODEL),
        "encoder_sha256": sha(MODEL),
        "features": FEATURES,
        "folds": 5,
        "fold_group": "SHA256(inchikey14) mod 5; known/unknown versions of a molecule share the same fold",
        "label": "historical first candidate is correct, never provided as an input feature",
        "actions": {
            "H": "historical hybrid",
            "U": "common-pool retrieval",
            "F": "historical-extended neural RRF .75",
            "N": "neural only",
        },
        "configs": policy_configs(),
        "selection": "maximum unknown OOF development MRR subject to known >= frozen large-model baseline - .001 and unknown >= baseline; simpler on ties",
        "acceptance": "fresh unknown paired CI95 lower > 0 and known MRR difference >= -.001",
        "fresh_molecules": 2000,
        "note": "OOF applies to router only; encoder was previously selected on development data. Fresh cohort excludes every previous query set.",
    }
    path = ROOT / "protocol.json"
    if path.exists() and json.loads(path.read_text()) != spec:
        raise ValueError("Protocol changed")
    write_json(path, spec)
    return spec


def prepare_probabilities(split):
    # Control generation owns <split>_probabilities.npy for the old encoder.
    path = ROOT / f"encoder_{split}_probabilities.npy"
    if path.exists():
        return np.load(path)
    from casmi_ml.data import cache_features

    model, checkpoint = load_deployment_checkpoint(MODEL, "scale")
    directory = OLD / "features/dev" if split == "dev" else ROOT / "features/fresh"
    if not (directory / "complete.json").exists():
        cache_features(
            pd.read_parquet(ROOT / "fresh.parquet"),
            checkpoint["preprocessing"],
            directory,
        )
    probs = predict_cpu(model, directory)
    np.save(path, probs)
    return probs


def build_rows(split, mode, ranker, frame, probs):
    out = ROOT / f"{split}_{mode}_rows.json"
    if out.exists():
        return json.loads(out.read_text())
    rows = []
    for i, r in enumerate(ranker.records):
        if mode == "known" and not r["known"]:
            continue
        key = r["key"]
        ids = ranker.groups[key]
        group = frame.iloc[ids]
        p = np.clip(probs[ids].mean(0).astype(np.float64), 1e-6, 1 - 1e-6)
        keys = r["available"]["union35"]
        fps = ranker.fps(keys)
        scores = fps @ (np.log(p) - np.log1p(-p)) / 2048
        order = sorted(range(len(keys)), key=lambda j: (-scores[j], keys[j]))
        neural = [keys[j] for j in order]
        h = r["rankings"]["coconut15"]
        u = r["current_full"]
        f = low_confidence_rank(h, u, neural, 0.75)
        actions = {"H": h[:25], "U": u[:25], "F": f[:25], "N": neural[:25]}
        histfirst = h[0] if h else None
        features = router_features(
            group, p, keys, fps, h, u, r["confidence"], r["margin"], center_mass(group)
        )
        base = "H" if r["confidence"] >= 0.5 else "F"
        rows.append(
            {
                "key": key,
                "mode": mode,
                "features": features,
                "actions": actions,
                "baseline_action": base,
                "rr": {name: rr(key, rank) for name, rank in actions.items()},
                "covered": key
                in set(r["available"]["union35"]) | set(r["available"]["coconut15"]),
                "label": int(histfirst == key),
            }
        )
        if (i + 1) % 250 == 0:
            print("router features", split, mode, i + 1, flush=True)
    write_json(out, rows)
    return rows


def classifier(name):
    if name == "logistic":
        return make_pipeline(
            StandardScaler(), LogisticRegression(C=0.3, max_iter=1500, random_state=42)
        )
    return HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.06,
        max_leaf_nodes=7,
        min_samples_leaf=30,
        l2_regularization=10,
        early_stopping=False,
        random_state=42,
    )


def chosen_actions(rows, config, probabilities=None):
    if config["type"] == "baseline":
        return [r["baseline_action"] for r in rows]
    values = (
        np.asarray([r["features"][0] for r in rows])
        if config["type"] == "fixed"
        else probabilities
    )
    return [
        config["high"] if p >= config["threshold"] else config["low"] for p in values
    ]


def score(rows, actions, mode):
    result = []
    for r, a in zip(rows, actions):
        if r["mode"] != mode:
            continue
        reciprocal = r["rr"][a]
        result.append(
            {
                "key": r["key"],
                "reciprocal_rank": reciprocal,
                "covered": int(r["covered"]),
                "top1": int(reciprocal == 1),
                "top5": int(reciprocal >= 0.2),
                "top25": int(reciprocal > 0),
            }
        )
    frame = pd.DataFrame(result)
    report = {
        "molecules": len(frame),
        "mrr25": float(frame.reciprocal_rank.mean()),
        "candidate_recall": float(frame.covered.mean()),
        **{n: float(frame[n].mean()) for n in ["top1", "top5", "top25"]},
    }
    return report, frame


def develop(spec):
    selection_path = ROOT / "selection.json"
    if selection_path.exists():
        return json.loads(selection_path.read_text())
    probs = prepare_probabilities("dev")
    rankers = dev_rankers()
    frame = pd.read_parquet(OLD / "dev.parquet")
    rows = []
    for mode in ["unknown", "known"]:
        rows.extend(build_rows("dev", mode, rankers[mode], frame, probs))
    x = np.asarray([r["features"] for r in rows])
    y = np.asarray([r["label"] for r in rows])
    folds = np.asarray(
        [
            int.from_bytes(hashlib.sha256(r["key"].encode()).digest()[:8], "big") % 5
            for r in rows
        ]
    )
    modes = np.asarray([r["mode"] for r in rows])
    predictions = {}
    for name in ["logistic", "boosted"]:
        path = ROOT / f"oof_{name}.npy"
        if path.exists():
            predictions[name] = np.load(path)
            continue
        pred = np.empty(len(rows))
        for fold in range(5):
            train = folds != fold
            test = ~train
            assert not {rows[i]["key"] for i in np.flatnonzero(train)} & {
                rows[i]["key"] for i in np.flatnonzero(test)
            }
            model = classifier(name)
            model.fit(x[train], y[train])
            pred[test] = model.predict_proba(x[test])[:, 1]
        np.save(path, pred)
        predictions[name] = pred
    base_actions = chosen_actions(rows, spec["configs"][0])
    baseline = {m: score(rows, base_actions, m)[0] for m in ["unknown", "known"]}
    comparisons = []
    for config in spec["configs"]:
        actions = chosen_actions(rows, config, predictions.get(config["type"]))
        reports = {m: score(rows, actions, m)[0] for m in ["unknown", "known"]}
        eligible = (
            reports["known"]["mrr25"] >= baseline["known"]["mrr25"] - 0.001
            and reports["unknown"]["mrr25"] >= baseline["unknown"]["mrr25"]
        )
        comparisons.append({"config": config, **reports, "eligible": eligible})
    complexity = {"baseline": 0, "fixed": 1, "logistic": 2, "boosted": 3}
    winner = max(
        (r for r in comparisons if r["eligible"]),
        key=lambda r: (r["unknown"]["mrr25"], -complexity[r["config"]["type"]]),
    )
    name = winner["config"]["type"]
    model_path = None
    if name in ["logistic", "boosted"]:
        model = classifier(name)
        model.fit(x, y)
        model_path = ROOT / "router.joblib"
        joblib.dump(model, model_path)
    selection = {
        "winner": winner,
        "baseline": baseline,
        "model_path": str(model_path) if model_path else None,
        "model_sha256": sha(model_path) if model_path else None,
        "protocol_sha256": sha(ROOT / "protocol.json"),
        "frozen_before_fresh_holdout": True,
    }
    write_json(ROOT / "dev_comparisons.json", comparisons)
    write_json(selection_path, selection)
    pd.DataFrame(
        {
            "key": [r["key"] for r in rows],
            "mode": modes,
            "fold": folds,
            "label": y,
            **{f"oof_{k}": v for k, v in predictions.items()},
        }
    ).to_csv(ROOT / "oof_predictions.csv", index=False)
    print("FROZEN ROUTER", json.dumps(selection), flush=True)
    return selection


def fresh_data():
    if (ROOT / "fresh.parquet").exists():
        return
    catalog = pd.read_parquet(OLD / "catalog.parquet")
    used = set()
    for directory, names in [
        (
            OLD,
            ["train", "dev", "holdout", "fresh_holdout", "diagnostic", "final_train"],
        ),
        (ABLATION, ["fresh"]),
        (SCALE, ["fresh", "train60k"]),
    ]:
        for name in names:
            used.update(
                pd.read_parquet(
                    directory / f"{name}.parquet", columns=["inchikey14"]
                ).inchikey14
            )
    keys = set(
        catalog.loc[
            (catalog.split == "holdout") & ~catalog.inchikey14.isin(used), "inchikey14"
        ].head(2000)
    )
    assert len(keys) == 2000 and not keys & used
    frame = sample_keys(keys, 20261001)
    frame.to_parquet(ROOT / "fresh.parquet", index=False)
    write_json(
        ROOT / "fresh_manifest.json",
        {
            "molecules": len(keys),
            "spectra": len(frame),
            "keys": sorted(keys),
            "overlap_with_prior_queries_and_training": 0,
            "selection_sha256": sha(ROOT / "selection.json"),
            "sha256": sha(ROOT / "fresh.parquet"),
        },
    )


def fresh_ranker(mode):
    import casmi_ml.ablation as controls
    from casmi_ml.ranking import build_candidates

    previous = controls.ROOT
    controls.ROOT = ROOT
    try:
        records = controls.build_records("fresh", mode)
    finally:
        controls.ROOT = previous
    frame = pd.read_parquet(ROOT / "fresh.parquet")
    catalog = pd.read_parquet(OLD / "catalog.parquet")
    if mode == "known":
        observed = set(
            pd.read_parquet(
                ROOT / "reference/fresh_known/rows.parquet", columns=["inchikey14"]
            ).inchikey14
        )
        catalog = catalog[
            ((catalog.split == "train") | catalog.inchikey14.isin(frame.inchikey14))
            & catalog.inchikey14.isin(observed)
        ]
    return Ranker(
        frame, records, build_candidates(catalog, COCONUT, final=mode == "known")
    )


def evaluate(selection):
    path = ROOT / "fresh_report.json"
    if path.exists():
        return json.loads(path.read_text())
    if selection["winner"]["config"]["type"] == "baseline":
        report = {
            "accepted": False,
            "reason": "No development improvement; no new holdout exposed",
        }
        write_json(path, report)
        return report
    fresh_data()
    probs = prepare_probabilities("fresh")
    frame = pd.read_parquet(ROOT / "fresh.parquet")
    model = joblib.load(selection["model_path"]) if selection["model_path"] else None
    reports = {}
    for mode in ["unknown", "known"]:
        ranker = fresh_ranker(mode)
        rows = build_rows("fresh", mode, ranker, frame, probs)
        probabilities = (
            model.predict_proba(np.asarray([r["features"] for r in rows]))[:, 1]
            if model
            else None
        )
        actions = chosen_actions(rows, selection["winner"]["config"], probabilities)
        report, new = score(rows, actions, mode)
        baseline, old = score(rows, chosen_actions(rows, {"type": "baseline"}), mode)
        report["baseline"] = baseline
        report["difference"] = bootstrap_difference(new, old)
        report["action_counts"] = pd.Series(actions).value_counts().to_dict()
        new.to_csv(ROOT / f"fresh_{mode}_selected.csv", index=False)
        old.to_csv(ROOT / f"fresh_{mode}_baseline.csv", index=False)
        reports[mode] = report
        del ranker
        gc.collect()
        print("FRESH ROUTER", mode, json.dumps(report), flush=True)
    reports["accepted"] = (
        reports["unknown"]["difference"]["ci95"][0] > 0
        and reports["known"]["difference"]["difference"] >= -0.001
    )
    reports["selection"] = selection["winner"]["config"]
    write_json(path, reports)
    return reports


def render(report):
    selection = json.loads((ROOT / "selection.json").read_text())
    comparisons = json.loads((ROOT / "dev_comparisons.json").read_text())
    text = "# 大模型路由校准实验（2026-09-29）\n\n"
    text += "编码器固定为上一轮 60K、10.7M 参数残差 MLP；CPU float32 推理。路由器只使用谱图/候选可观测特征，已知/未知场景标记、真实结构和候选覆盖标签均不作为输入。\n\n"
    text += "五折按分子分组，同一分子的已知/未知参考谱模拟落在同一折。编码器已在开发集选过模型，所以这里的 OOF 仅是路由器的 OOF；最终结论以另一批全新分子的验收为准。\n\n"
    text += f"冻结方案：`{selection['winner']['config']}`。独立验收通过：**{report['accepted']}**。\n\n"
    text += "| 开发 OOF 方案 | 未知 MRR | 已知 MRR | 合格 |\n|---|---:|---:|---|\n"
    best = sorted(comparisons, key=lambda r: r["unknown"]["mrr25"], reverse=True)[:12]
    for r in [comparisons[0]] + best:
        text += f"| {r['config']['name']} | {r['unknown']['mrr25']:.6f} | {r['known']['mrr25']:.6f} | {r['eligible']} |\n"
    if "unknown" in report:
        text += "\n| 全新留出场景 | 分子数 | 固定阈值大模型 | 新路由 | 差值 95% CI |\n|---|---:|---:|---:|---|\n"
        for mode in ["unknown", "known"]:
            r = report[mode]
            text += f"| {mode} | {r['molecules']} | {r['baseline']['mrr25']:.6f} | {r['mrr25']:.6f} | {r['difference']['ci95']} |\n"
    text += "\nH=历史 hybrid，U=共同候选池检索，F=神经 RRF 0.75，N=纯神经排序。完整 53 组配置在 dev_comparisons.json。\n\n"
    text += "验收规则：未知谱改善的 paired bootstrap 95% 区间下限 >0，已知谱点估计下降不超过 0.001。若失败保留上一轮配置，不用本轮留出集重新选模型。尚未提交 Kaggle。\n"
    if "known" in report:
        text += "\n已知谱图差值区间跨零，因此通过的是预先规定的点估计保护门槛，并非统计上证明已知谱无损或有提升。未知/已知场景的分布也不等于比赛隐藏测试集。上一轮大模型与本轮路由使用不同留出分子，不能直接相加两轮绝对收益。\n"
    text += "\n通过验收后的本地配置：`artifacts/router_20260929/deployment_recipe.json`。使用 `python -m casmi_ml.secondary_inference --recipe artifacts/router_20260929/deployment_recipe.json --output artifacts/router_20260929/submission.csv` 复现。路由特征共 17 个；实验与推理共用 router_features.py。模型依赖版本见 requirements-ml.txt。\n"
    (ROOT / "REPORT.md").write_text(text)
    Path("docs/ROUTER_20260929.md").write_text(text)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["develop", "evaluate", "run"])
    args = parser.parse_args()
    configure(42, 4)
    spec = protocol()
    selection = develop(spec)
    if args.stage != "develop":
        render(evaluate(selection))


if __name__ == "__main__":
    main()
