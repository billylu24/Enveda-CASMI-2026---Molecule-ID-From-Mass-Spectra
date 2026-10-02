import copy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from rdkit import Chem
from rdkit.Chem import Descriptors
import torch

from baseline import PROTON
from casmi_ml.adversarial import (
    BITS, ConditionalDiscriminator, ConditionalGenerator, alternating_step,
    canonical_structure, condition_features, load_generator, predict_fingerprints,
    prepare, soft_tanimoto_scores, split_identity, straight_through_bernoulli, train_pair,
)


def raw_key(smiles):
    return Chem.MolToInchiKey(Chem.MolFromSmiles(smiles))[:14]


def paired_row(smiles, source="library", row=0):
    mass = Descriptors.ExactMolWt(Chem.MolFromSmiles(smiles))
    return {"inchikey14": raw_key(smiles), "normalized_smiles": smiles,
            "ingest_lib": source, "precursor_mz": mass + PROTON, "adduct": "[M+H]+",
            "ionization_mode": "positive", "instrument_type": "fixture-instrument",
            "collision_energy_ev": [10., 20.], "precursor_error_ppm": 0.,
            "spectrum_id": str(row), "ms2_mzs": [43., 45. + row % 3],
            "ms2_normalized_intensities": [1., .5], "huge_embedding": [0.] * 8}


@pytest.fixture
def corpus(tmp_path):
    rows = [paired_row("C" * atoms, row=atoms) for atoms in range(1, 81)]
    # These different raw keys are one official tautomer identity.
    rows += [paired_row("CC(=O)C", "enveda-np-examples"), paired_row("CC(O)=C", "other")]
    rows += [paired_row("CC(=O)O", "other"), paired_row("CC(O)=O", "other")]
    path = tmp_path / "train.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


def test_canonical_identity_split_excludes_aliases_and_keeps_raw_mapping(corpus, tmp_path):
    root = tmp_path / "prepared"
    manifest = prepare(corpus, root, train_limit=80, eval_limit=80,
                       exclude_raw_keys=[raw_key("CC(=O)O")])
    catalog = pd.read_parquet(root / "catalog.parquet")
    ketone = canonical_structure("CC(=O)C")[0]
    assert ketone == canonical_structure("CC(O)=C")[0]
    assert set(catalog.loc[catalog.identity.eq(ketone), "split"]) == {"excluded"}
    assert {raw_key("CC(=O)C"), raw_key("CC(O)=C")} <= set(manifest["excluded_rawkeys"])
    assert raw_key("CC(=O)O") in manifest["excluded_rawkeys"]
    split_sets = []
    for split in ["train", "dev", "acceptance"]:
        frame = pd.read_parquet(root / f"{split}.parquet")
        identities = set(frame.identity)
        split_sets.append(identities)
        assert all(split_identity(identity)[0] == split for identity in identities)
        assert "huge_embedding" not in frame.columns
        assert all(len(value) == BITS // 8 for value in frame.fingerprint)
        assert set(frame.inchikey14) <= set(catalog.inchikey14)
    assert not (split_sets[0] & split_sets[1] or split_sets[0] & split_sets[2] or split_sets[1] & split_sets[2])
    assert prepare(corpus, root, train_limit=80, eval_limit=80,
                   exclude_raw_keys=[raw_key("CC(=O)O")]) == manifest
    with pytest.raises(ValueError, match="configuration changed"):
        prepare(corpus, root, train_limit=79, eval_limit=80,
                exclude_raw_keys=[raw_key("CC(=O)O")])


def test_visible_test_truth_used_only_when_explicit(corpus, tmp_path):
    unlabeled = tmp_path / "unlabeled.parquet"
    pd.DataFrame([{"molecule_id": "query", "precursor_mz": 100.}]).to_parquet(unlabeled)
    report = prepare(corpus, tmp_path / "unlabeled_split", train_limit=80, eval_limit=80, test_path=unlabeled)
    assert report["visible_test_exclusion"]["status"] == "unlabeled_test_identities_not_identifiable"
    labeled = tmp_path / "labeled.parquet"
    pd.DataFrame([{"normalized_smiles": "CCCCC"}]).to_parquet(labeled)
    report = prepare(corpus, tmp_path / "labeled_split", train_limit=80, eval_limit=80, test_path=labeled)
    assert report["visible_test_exclusion"]["status"] == "explicit_labels_used"
    assert canonical_structure("CCCCC")[0] in report["excluded_identities"]


def test_unlabeled_visible_copy_excludes_entire_identity_and_raw_aliases(corpus, tmp_path):
    training = pd.read_parquet(corpus)
    # Use an otherwise eligible identity with a unique vector, plus another
    # spectrum under the same identity that must also be excluded.
    target = paired_row("CCNCC", row=999)
    target["ms2_mzs"], target["ms2_normalized_intensities"] = [67.1234, 78.5678], [1., .123]
    repeat = {**target, "ms2_mzs": [50., 85.], "spectrum_id": "another-source-copy"}
    pd.concat([training, pd.DataFrame([target, repeat])], ignore_index=True).to_parquet(corpus, index=False)
    test = tmp_path / "unlabeled_spectra.parquet"
    pd.DataFrame([{name: value for name, value in target.items()
                   if name in {"ms2_mzs", "ms2_normalized_intensities", "precursor_mz", "adduct"}}]).to_parquet(test)
    root = tmp_path / "content_split"
    report = prepare(corpus, root, train_limit=80, eval_limit=80, test_path=test)
    official = canonical_structure("CCNCC")[0]
    assert official in report["visible_spectrum_exclusion"]["matched_official_identities"]
    assert target["inchikey14"] in report["excluded_rawkeys"]
    assert set(pd.read_parquet(root / "catalog.parquet").loc[
        lambda frame: frame.identity.eq(official), "split"]) == {"excluded"}
    for name in ["train", "dev", "acceptance"]:
        assert official not in set(pd.read_parquet(root / f"{name}.parquet").identity)


def test_alternating_updates_both_networks_and_supervised_control_leaves_d_fixed():
    torch.set_num_threads(1)
    torch.manual_seed(7)
    generator = ConditionalGenerator(12, width=16, noise_dim=4, bits=24)
    discriminator = ConditionalDiscriminator(12, width=16, bits=24)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=.001)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=.001)
    conditions = torch.randn(8, 12)
    target = (torch.rand(8, 24) > .8).float()
    identities = torch.arange(8)
    old_g = [parameter.detach().clone() for parameter in generator.parameters()]
    old_d = [parameter.detach().clone() for parameter in discriminator.parameters()]
    stats = alternating_step(generator, discriminator, g_optimizer, d_optimizer,
                             conditions, target, identities, adversarial_weight=.05)
    assert stats["d_updates"] == stats["g_updates"] == 1
    assert all(np.isfinite(value) for value in stats.values() if value is not None)
    assert any(not torch.equal(before, after) for before, after in zip(old_g, generator.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(old_d, discriminator.parameters()))
    old_d = [parameter.detach().clone() for parameter in discriminator.parameters()]
    stats = alternating_step(generator, discriminator, g_optimizer, d_optimizer,
                             conditions, target, identities, adversarial_weight=0.)
    assert stats["d_updates"] == 0 and stats["g_updates"] == 1 and stats["d_loss"] is None
    assert all(torch.equal(before, after) for before, after in zip(old_d, discriminator.parameters()))


def test_hard_fake_bits_pass_generator_gradients_and_same_identity_is_not_mismatch():
    probability = torch.full((4, 8), .25, requires_grad=True)
    hard = straight_through_bernoulli(probability)
    assert torch.all((hard == 0) | (hard == 1))
    hard.sum().backward()
    assert torch.equal(probability.grad, torch.ones_like(probability))
    # Parentheses keep the forward bits exactly binary for arbitrary floats.
    assert torch.all((straight_through_bernoulli(torch.rand(64, 32)) % 1) == 0)
    generator = ConditionalGenerator(6, 8, 3, 8)
    discriminator = ConditionalDiscriminator(6, 8, 8)
    stats = alternating_step(generator, discriminator, torch.optim.Adam(generator.parameters()),
                             torch.optim.Adam(discriminator.parameters()), torch.randn(4, 6),
                             torch.zeros(4, 8), torch.zeros(4, dtype=torch.int64))
    assert stats["d_mismatch_accuracy"] is None


def test_fixed_generator_noise_is_identical_after_discriminator_rng_consumption():
    torch.manual_seed(33)
    original = ConditionalGenerator(6, 12, 4, 16)
    baseline, cgan = copy.deepcopy(original), copy.deepcopy(original)
    d_baseline = ConditionalDiscriminator(6, 12, 16)
    d_cgan = copy.deepcopy(d_baseline)
    condition, target, identities = torch.randn(5, 6), (torch.rand(5, 16) > .8).float(), torch.arange(5)
    fixed_noise = torch.randn(5, 4)
    received = {"baseline": [], "cgan": []}
    baseline.register_forward_pre_hook(lambda module, args: received["baseline"].append(args[1].detach().clone()))
    cgan.register_forward_pre_hook(lambda module, args: received["cgan"].append(args[1].detach().clone()))
    control = alternating_step(baseline, d_baseline, torch.optim.Adam(baseline.parameters()),
                               torch.optim.Adam(d_baseline.parameters()), condition, target, identities,
                               adversarial_weight=0., generator_noise=fixed_noise)
    adversarial = alternating_step(cgan, d_cgan, torch.optim.Adam(cgan.parameters()),
                                   torch.optim.Adam(d_cgan.parameters()), condition, target, identities,
                                   adversarial_weight=.05, generator_noise=fixed_noise)
    assert len(received["baseline"]) == 1 and len(received["cgan"]) == 2
    assert torch.equal(received["baseline"][-1], fixed_noise)
    assert torch.equal(received["cgan"][-1], fixed_noise)
    assert control["supervised_bce"] == adversarial["supervised_bce"]


def test_real_small_training_checkpoints_and_inference_are_order_stable(corpus, tmp_path):
    root, output = tmp_path / "prepared", tmp_path / "trained"
    prepare(corpus, root, train_limit=40, eval_limit=8)
    # Acceptance may not even exist during training. Its labels cannot select
    # an epoch or accidentally enter feature preprocessing.
    acceptance = root / "acceptance.parquet"
    acceptance.rename(root / "acceptance_hidden.parquet")
    config = {"width": 16, "noise_dim": 4, "epochs": 3, "warmup_epochs": 1,
              "batch_size": 16, "threads": 1, "prediction_samples": 3}
    report = train_pair(root, output, config, seconds_per_arm=30, device="cpu")
    assert report["acceptance_used"] is False
    assert report["arms"]["baseline"]["d_steps_total"] == 0
    assert report["arms"]["cgan"]["selected_d_steps"] > 0
    model, checkpoint = load_generator(output / "cgan" / "model.pt")
    assert checkpoint["arm"] == "cgan" and checkpoint["d_steps"] > 0
    assert checkpoint["epoch"] > config["warmup_epochs"]
    queries = pd.read_parquet(root / "dev.parquet").iloc[:4].reset_index(drop=True)
    preprocessing = checkpoint["preprocessing"]
    predictions = predict_fingerprints(model, queries, preprocessing, samples=3, batch_size=1)
    reverse = predict_fingerprints(model, queries.iloc[::-1], preprocessing, samples=3, batch_size=3)[::-1]
    assert predictions.shape == (len(queries), BITS)
    np.testing.assert_allclose(predictions, reverse, atol=1e-7, rtol=1e-6)
    assert np.isfinite(predictions).all() and ((predictions >= 0) & (predictions <= 1)).all()
    assert predict_fingerprints(model, queries.iloc[:0], preprocessing).shape == (0, BITS)
    assert condition_features(queries, preprocessing).shape[1] == checkpoint["condition_dim"]
    assert train_pair(root, output, config, seconds_per_arm=30, device="cpu")["arms"] == report["arms"]
    history = json.loads((output / "cgan" / "history.json").read_text())
    assert all("bce_gap" in epoch and "noise_probability_variance" in epoch for epoch in history)
    baseline_history = json.loads((output / "baseline" / "history.json").read_text())
    assert baseline_history[0]["dev_bce"] == history[0]["dev_bce"]
    assert baseline_history[0]["supervised_bce"] == history[0]["supervised_bce"]
    assert report["matched_generator_training_noise"] is True


def test_fixed_soft_tanimoto_ranking_and_empty_pool():
    probabilities = np.zeros(BITS, np.float32)
    probabilities[:2] = [.8, .9]
    candidates = np.zeros((2, BITS), np.uint8)
    candidates[0, :2] = 1
    candidates[1, 100] = 1
    scores = soft_tanimoto_scores(probabilities, candidates)
    assert scores[0] > scores[1] == 0
    assert soft_tanimoto_scores(probabilities, np.empty((0, BITS))).shape == (0,)
    with pytest.raises(ValueError, match="Non-finite"):
        soft_tanimoto_scores(np.full(BITS, np.nan), candidates)


def test_rejects_budget_that_never_reaches_actual_adversarial_training(corpus, tmp_path):
    root = tmp_path / "prepared"
    prepare(corpus, root, train_limit=10, eval_limit=3)
    with pytest.raises(ValueError, match="configuration"):
        train_pair(root, tmp_path / "invalid", {"epochs": 2, "warmup_epochs": 2})
