"""Training-only audit of connected single-bond fragments for frozen12 hypotheses."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem

from casmi_ml.candidate_chemistry_specificity_audit import CACHE
from casmi_ml.data import write_json
from casmi_ml.direct_experiment import train_data
from casmi_ml.encoder_pair_finetuning import HARD, validate_hard_table
from casmi_ml.extended_diagnostic_rules import RULES
from casmi_ml.metfrag import digest
from casmi_ml.research_loop import Controller
from casmi_ml.research_protocol import ROOT, TRAIN, freeze
from casmi_ml.strict_mass_negative_mask import admissible_negatives
from casmi_ml.training_query_pairs import paired_rows

# Heavy atom compositions only; H transfer, charge localization, rearrangement,
# collision energy and competing paths are deliberately not inferred.
COMPOSITIONS = (
    {"C": 6},
    {"C": 7},
    {"C": 7, "O": 1},
    {"C": 7, "O": 2},
    {"P": 1, "O": 3},
    {"P": 1, "O": 4},
    {"C": 1, "O": 1},
    {"S": 1, "O": 2},
    {"S": 1, "O": 3},
    {"P": 1, "O": 4},
    {"P": 1, "O": 3},
    {"C": 2, "O": 2},
)
TARGETS = tuple(tuple(sorted(c.items())) for c in COMPOSITIONS)


def connected_support(smiles):
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("Valid candidate molecule required")
    if len(Chem.GetMolFrags(molecule)) != 1:
        return (False,) * len(TARGETS)
    adjacency = [[] for _ in molecule.GetAtoms()]
    symbols = [a.GetSymbol() for a in molecule.GetAtoms()]
    for bond in molecule.GetBonds():
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        adjacency[a].append(b)
        adjacency[b].append(a)
    total = Counter(s for s in symbols if s != "H")
    available = set()
    for bond in molecule.GetBonds():
        if bond.IsInRing() or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        reached, stack = {a}, [a]
        while stack:
            at = stack.pop()
            for other in adjacency[at]:
                if (at == a and other == b) or (at == b and other == a):
                    continue
                if other not in reached:
                    reached.add(other)
                    stack.append(other)
        if b in reached:
            raise ValueError("Non-ring single bond unexpectedly has an alternate path")
        side = Counter(symbols[i] for i in reached if symbols[i] != "H")
        for part in (side, total - side):
            available.add(tuple(sorted(part.items())))
    return tuple(target in available for target in TARGETS)


def pair_summary(observed, broad, specific, hard, compatible):
    if observed.shape != (len(broad), len(RULES) * 3):
        raise ValueError("Aligned frozen36 observed channels required")
    if broad.shape != specific.shape or broad.shape[1] != len(RULES):
        raise ValueError("Aligned12 broad and connected support arrays required")
    refined = specific & broad
    channels = []
    for i, rule in enumerate(RULES):
        for k, kind in enumerate(("ion", "precursor_loss", "peak_loss")):
            use = compatible & (observed[:, i * 3 + k] > 0)[:, None]
            tied = use & (broad[:, i, None] == broad[hard, i])
            delta = refined[:, i, None].astype(np.int8) - refined[hard, i]
            channels.append(
                {
                    "channel": rule.name + ":" + kind,
                    "observed_compatible_pairs": int(use.sum()),
                    "broad_ties": int(tied.sum()),
                    "broad_ties_separated": int((tied & (delta != 0)).sum()),
                    "separated_ties_positive_advantage": int(
                        (tied & (delta > 0)).sum()
                    ),
                    "separated_ties_negative_advantage": int(
                        (tied & (delta < 0)).sum()
                    ),
                    "refined_positive_advantage": int((use & (delta > 0)).sum()),
                    "refined_negative_advantage": int((use & (delta < 0)).sum()),
                }
            )
    alias = compatible & (broad[:, None, :] == broad[hard]).all(-1)
    separated = alias & (refined[:, None, :] != refined[hard]).any(-1)
    return {
        "all12_broad_alias_pairs": int(alias.sum()),
        "aliases_separated_by_connected_refinement": int(separated.sum()),
        "unique_broad_patterns": len(np.unique(broad, axis=0)),
        "unique_refined_patterns": len(np.unique(refined, axis=0)),
        "channels": channels,
    }


def run(output):
    c = Controller()
    if not c.local_only():
        raise ValueError("Connected-fragment audit requires local-only mode")
    output = Path(output)
    cached = json.loads((CACHE / "report.json").read_text())
    protocol = json.loads((CACHE / "protocol.json").read_text())
    for field, source in [
        ("training_sha256", TRAIN),
        ("negative_table_sha256", HARD),
        ("feature_sha256", "casmi_ml/extended_diagnostic_rules.py"),
        ("chemistry_sha256", "casmi_ml/chemistry.py"),
    ]:
        if protocol[field] != digest(source):
            raise ValueError("Frozen training feature binding changed: " + field)
    for name, sha in cached["private_input_sha256"].items():
        if digest(CACHE / name) != sha:
            raise ValueError("Frozen private training input changed")
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "training_sha256": digest(TRAIN),
            "negative_table_sha256": digest(HARD),
            "cache_report_sha256": digest(CACHE / "report.json"),
            "rule_source_sha256": digest("casmi_ml/extended_diagnostic_rules.py"),
            "heavy_compositions": dict(zip([r.name for r in RULES], COMPOSITIONS)),
            "rule": "All original isolated60K RNG42 representatives/pinned15/same-ion partner bindings. Refinement=original broad SMARTS AND heavy composition of either connected side of exactly one non-ring single-bond cut. Disconnected molecules produce no refinement. Reuse only source-bound305 training36 mean channels. No model, dev score, absent-feature penalty, H/charge/energy/rearrangement inference or deployment.",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    started = time.monotonic()
    train, data = pd.read_parquet(TRAIN), train_data()
    groups = train.groupby("inchikey14", sort=True).indices
    devkeys = set(
        pd.read_parquet(ROOT / "researchdev.parquet", columns=["inchikey14"]).inchikey14
    )
    if len(groups) != 60000 or list(groups) != data["keys"] or set(groups) & devkeys:
        raise ValueError("Original isolated training identities required")
    rng = np.random.default_rng(42)
    ids = np.array([rng.choice(v) for v in groups.values()])
    rows = pd.read_parquet(CACHE / "training_rows.parquet")
    if (
        list(rows.inchikey14) != data["keys"]
        or not np.array_equal(rows.representative_row, ids)
        or not np.array_equal(rows.paired_row, paired_rows(train, ids))
    ):
        raise ValueError("Cached train representative/partner identities differ")
    hard = np.load(HARD)
    validate_hard_table(hard, data)
    supports = []
    smiles = train.iloc[ids].normalized_smiles.to_numpy()
    for start in range(0, len(smiles), 1024):
        if c.stopped():
            return None
        supports.extend(connected_support(s) for s in smiles[start : start + 1024])
        if start % 10240 == 0:
            print(
                "connected-fragment training audit", start, "/", len(smiles), flush=True
            )
    specific = np.asarray(supports, dtype=bool)
    broad = np.load(CACHE / "candidate_support.npy").astype(bool)
    observed = np.load(CACHE / "mean_chemical.npy")
    report = pair_summary(
        observed, broad, specific, hard, admissible_negatives(data["masses"], hard)
    )
    np.save(output / "private_connected_support.npy", specific)
    report.update(
        diagnostic_only=True,
        local_only=True,
        independent_acceptance=False,
        training_molecules=len(ids),
        train_development_overlap=0,
        disconnected_training_molecules=int(sum("." in s for s in smiles)),
        private_support_sha256=digest(output / "private_connected_support.npy"),
        seconds=time.monotonic() - started,
        scope="Training-only heavy atom connectivity refinement availability/association. Directional contrasts are observational, not a chemical mechanism or predictive gain. No best/model update; no new holdout or publication.",
    )
    write_json(output / "report.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    print(json.dumps(run(parser.parse_args().output), indent=2))


if __name__ == "__main__":
    main()
