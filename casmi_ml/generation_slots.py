"""Full development comparison of explicit generated-candidate insertion slots."""

import argparse
import fcntl
import json
from pathlib import Path

from casmi_ml.data import write_json
from casmi_ml.generation_experiment import generate
from casmi_ml.metfrag import digest
from casmi_ml.ranking import metrics
from casmi_ml.research_protocol import ROOT, freeze


def insert_generated(base, generated, prefix, slots):
    if prefix < 1 or slots < 1:
        raise ValueError("Positive insertion prefix and slots required")
    novel = [key for key in dict.fromkeys(generated) if key not in set(base)][:slots]
    return base[:prefix] + novel + base[prefix:]


def run(output, incumbent, variant, source=ROOT):
    output, incumbent, source = Path(output), Path(incumbent), Path(source)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "version": 1,
            "source": str(source),
            "development_sha256": digest(source / "researchdev.parquet"),
            "incumbent_report_sha256": digest(incumbent / "report.json"),
            "incumbent_variant": variant,
            "samples": 128,
            "prefixes": [5, 10, 20],
            "slots": [1, 3, 5],
            "holdout_used": False,
        },
    )
    gpu = Path("artifacts/research_loop/gpu.lock")
    gpu.parent.mkdir(parents=True, exist_ok=True)
    with gpu.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        generated = generate(source, samples=128)
    by_key = {r["key"]: [c["key"] for c in r["candidates"]] for r in generated}
    if len(by_key) < 2000:
        raise ValueError("Full 2000-molecule generation required")
    variants = {
        "baseline": None,
        **{f"slots_{p}_{n}": (p, n) for p in [5, 10, 20] for n in [1, 3, 5]},
    }
    report = {}
    for mode in ["unknown", "known"]:
        records = json.loads((incumbent / f"{mode}_records.json").read_text())
        confidence = {
            r["key"]: r["confidence"]
            for r in json.loads(
                (source / f"researchdev_{mode}_chemical.json").read_text()
            )
        }
        chosen = [r for r in records if mode == "unknown" or r["known"]]
        if set(by_key) != {r["key"] for r in records}:
            raise ValueError("Generated and retrieval cohort keys differ")
        report[mode] = {}
        for name, spec in variants.items():
            rankings, pools = {}, {}
            for row in chosen:
                key = row["key"]
                base = row["variants"][variant]["ranking"]
                rankings[key] = (
                    base
                    if spec is None or confidence[key] >= 0.5
                    else insert_generated(base, by_key[key], *spec)
                )
                pools[key] = row["variants"][variant]["pool"] + (
                    [] if spec is None else by_key[key]
                )
            result, per = metrics(rankings, pools)
            report[mode][name] = result
            per.to_csv(output / f"{mode}_{name}.csv", index=False)
    write_json(output / "report.json", report)
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--incumbent", required=True, type=Path)
    p.add_argument("--variant", required=True)
    a = p.parse_args()
    print(json.dumps(run(a.output, a.incumbent, a.variant), indent=2))


if __name__ == "__main__":
    main()
