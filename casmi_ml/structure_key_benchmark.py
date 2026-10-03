"""Exact bounded SMILES identity cache benchmark over every baseline handoff."""

import argparse
import json
import time
from pathlib import Path

from rdkit import Chem

from casmi_ml.cached_structure_keys import structure_key
from casmi_ml.data import write_json
from casmi_ml.low_fragment_first import CONTROL
from casmi_ml.metfrag import digest
from casmi_ml.research_protocol import freeze


def run(output):
    output = Path(output)
    freeze(
        output / "protocol.json",
        {
            "source_sha256": digest(__file__),
            "cache_source_sha256": digest("casmi_ml/cached_structure_keys.py"),
            "input_sha256": {
                mode: digest(CONTROL / mode / "baseline_full.json")
                for mode in ("unknown", "known")
            },
            "scope": "Every original handoff SMILES in actual group/list order; original parse+connectivity key vs bounded100K pure memoization, no declaration-derived identity or model changes",
            "local_only": True,
            "independent_acceptance": False,
        },
    )
    reports = {}
    for mode in ["unknown", "known"]:
        records = json.loads((CONTROL / mode / "baseline_full.json").read_text())
        strings = [s for record in records for s in record["smiles"]]
        started = time.monotonic()
        expected = [Chem.MolToInchiKey(Chem.MolFromSmiles(s))[:14] for s in strings]
        legacy_seconds = time.monotonic() - started
        structure_key.cache_clear()
        started = time.monotonic()
        actual = [structure_key(s) for s in strings]
        cached_seconds = time.monotonic() - started
        if actual != expected:
            raise ValueError("Cached actual SMILES key differs")
        reports[mode] = {
            "input_strings": len(strings),
            "groups": len(records),
            "all_actual_structure_keys_exact": True,
            "legacy_seconds": legacy_seconds,
            "cached_seconds": cached_seconds,
            "cache_info": structure_key.cache_info()._asdict(),
        }
        print(mode, reports[mode], flush=True)
    reports.update(
        diagnostic_only=True,
        local_only=True,
        independent_acceptance=False,
        activated=False,
        scope="Identity conversion segment only; full3340 inference control and source amendment required for activation",
    )
    write_json(output / "report.json", reports)
    return reports


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", required=True)
    print(json.dumps(run(**vars(p.parse_args())), indent=2))


if __name__ == "__main__":
    main()
