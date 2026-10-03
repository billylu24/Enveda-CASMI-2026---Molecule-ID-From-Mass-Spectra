"""Aggregate observable evidence coverage; no label-based feature or group selection."""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from casmi_ml.chemistry import RULES
from casmi_ml.data import write_json
from casmi_ml.diagnostic_residual import KINDS, evidence_features
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import ROOT, TRAIN, freeze


def summarize(features, groups):
    presence = features > 0
    active = presence.any(1)
    channels = {
        f"{rule.name}:{kind}": int(presence[:, i * 3 + j].sum())
        for i, rule in enumerate(RULES)
        for j, kind in enumerate(KINDS)
    }
    informative = presence[
        :,
        [
            i * 3 + j
            for i, rule in enumerate(RULES)
            if rule.name not in ("water", "carbon_monoxide")
            for j in range(3)
        ],
    ].any(1)
    molecules = {
        str(key): bool(active[indexes].any()) for key, indexes in groups.items()
    }
    molecule_informative = {
        str(key): bool(informative[indexes].any()) for key, indexes in groups.items()
    }
    return {
        "spectra": len(features),
        "spectra_with_evidence": int(active.sum()),
        "molecules": len(groups),
        "molecules_with_evidence": sum(molecules.values()),
        "spectra_with_evidence_other_than_water_or_CO": int(informative.sum()),
        "molecules_with_evidence_other_than_water_or_CO": sum(
            molecule_informative.values()
        ),
        "nonempty_channel_counts": channels,
        "strength_quantiles_when_nonempty": np.quantile(
            features.max(1)[active], [0.1, 0.5, 0.9]
        ).tolist()
        if active.any()
        else [],
        "active_channels": int(presence.any(0).sum()),
    }


def run(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "development_sha256": digest(ROOT / "researchdev.parquet"),
            "rule": "Matched original60K RNG42 representative spectra and complete development spectra; chemical features and observable coverage only. No label-selected groups, no accuracy or deployment claim.",
        },
    )
    columns = [
        "inchikey14",
        "ms2_mzs",
        "ms2_normalized_intensities",
        "precursor_mz",
        "ionization_mode",
        "instrument_type",
        "adduct",
    ]
    train = pd.read_parquet(TRAIN, columns=columns)
    rng = np.random.default_rng(42)
    ids = np.array(
        [rng.choice(v) for v in train.groupby("inchikey14", sort=True).indices.values()]
    )
    chosen = train.iloc[ids].reset_index(drop=True)
    del train
    dev = pd.read_parquet(ROOT / "researchdev.parquet", columns=columns)
    result = {}
    for name, frame in [
        ("train_representative", chosen),
        ("development_all_spectra", dev),
    ]:
        result[name] = {}
        arms = {}
        for precise in [False, True]:
            features = np.stack(
                [evidence_features(row, precise) for row in frame.to_dict("records")]
            )
            arms[precise] = features
            result[name]["precise" if precise else "coarse"] = summarize(
                features, frame.groupby("inchikey14", sort=True).indices
            )
        result[name]["coarse_only_nonempty_spectra"] = int(
            ((arms[False].max(1) > 0) & (arms[True].max(1) == 0)).sum()
        )
        print(name, result[name], flush=True)
    result.update(diagnostic_only=True, local_only=True, independent_acceptance=False)
    write_json(output / "report.json", result)
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
