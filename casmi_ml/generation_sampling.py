"""Label-independent per-query sampling identity shared by evaluation and inference."""

import hashlib
import json

from casmi_ml.ranking import spectrum_signature


def sampling_seed(group):
    rows = []
    for row in group.to_dict("records"):
        rows.append(
            {
                "adduct": str(row["adduct"]),
                "precursor_mz": float(row["precursor_mz"]),
                "spectrum": spectrum_signature(
                    row["ms2_mzs"], row["ms2_normalized_intensities"]
                ).hex(),
            }
        )
    value = json.dumps(
        sorted(rows, key=lambda r: json.dumps(r, sort_keys=True)), sort_keys=True
    )
    return int.from_bytes(hashlib.sha256(value.encode()).digest()[:8], "big") % (
        2**63 - 1
    )


def condition_for_group(encoder, preprocessing, group, device, include_mass=True):
    import numpy as np
    import torch

    from casmi_ml.chemistry import neutral_mass
    from casmi_ml.data import features

    values = [features(r, preprocessing) for r in group.to_dict("records")]
    batch = {
        k: torch.from_numpy(np.stack([v[i] for v in values])).to(device)
        for i, k in enumerate(["hist", "loss", "meta", "peaks", "mask"])
    }
    with torch.inference_mode():
        z = encoder.encoder(
            torch.cat([batch["hist"], batch["meta"], batch["loss"]], -1)
        )
        condition = torch.cat([z, batch["meta"]], -1).mean(0, keepdim=True)
        if include_mass:
            masses = [neutral_mass(r) for r in group.to_dict("records")]
            mass_features = np.array(
                [
                    [m / 1250 if m is not None else 0.0, float(m is None)]
                    for m in masses
                ],
                dtype=np.float32,
            )
            condition = torch.cat(
                [
                    condition,
                    torch.from_numpy(mass_features.mean(0, keepdims=True)).to(device),
                ],
                1,
            )
    return condition
