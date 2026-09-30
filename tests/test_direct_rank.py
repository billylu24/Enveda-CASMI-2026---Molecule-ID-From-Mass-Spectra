import unittest

import numpy as np
import pandas as pd
import torch

from casmi_ml.data import fingerprint, fit_preprocessing
from casmi_ml.direct_experiment import single_query
from casmi_ml.direct_models import (
    DirectRanker,
    MolecularGraph,
    batch_graphs,
    graph,
    score_group,
)
from casmi_ml.scale_models import ScaleModel


class DirectRankTests(unittest.TestCase):
    def test_multi_spectrum_scores_equal_mean_of_individual_matches(self):
        torch.manual_seed(42)
        frame = pd.DataFrame(
            [
                {
                    "precursor_mz": 101.0,
                    "ms2_mzs": [30.0, 60.0],
                    "ms2_normalized_intensities": [1.0, 0.5],
                    "collision_energy_ev": [10.0],
                    "adduct": "[M+H]+",
                    "ionization_mode": "positive",
                    "instrument_type": "test",
                },
                {
                    "precursor_mz": 101.0,
                    "ms2_mzs": [40.0, 80.0],
                    "ms2_normalized_intensities": [0.5, 1.0],
                    "collision_energy_ev": [40.0],
                    "adduct": "[M+H]+",
                    "ionization_mode": "positive",
                    "instrument_type": "test",
                },
            ]
        )
        prep = fit_preprocessing(frame)
        dim = 6 + sum(len(v) + 1 for v in prep["categories"].values())
        encoder = ScaleModel("wide_enhanced", dim).eval()
        ranker = DirectRanker("graph").eval()
        candidates = pd.DataFrame({"normalized_smiles": ["CCO", "CCC", "CCN"]})
        fps = np.stack([fingerprint(s) for s in candidates.normalized_smiles]).astype(
            np.float32
        )
        together = score_group(ranker, encoder, frame, prep, candidates, fps)
        separate = np.mean(
            [
                score_group(ranker, encoder, frame.iloc[[i]], prep, candidates, fps)
                for i in range(len(frame))
            ],
            axis=0,
        )
        np.testing.assert_allclose(together, separate, atol=2e-6, rtol=2e-6)

    def test_graph_atom_permutation_and_batch_independence(self):
        torch.manual_seed(42)
        model = MolecularGraph().eval()
        a, e, b = graph("CC(=O)Oc1ccccc1C(=O)O")
        permutation = np.arange(len(a))[::-1].copy()
        inverse = np.argsort(permutation)
        swapped = (a[permutation], inverse[e], b)
        original = model(batch_graphs([(a, e, b)]))
        reordered = model(batch_graphs([swapped, graph("[Na+]")]))[:1]
        torch.testing.assert_close(original, reordered, atol=2e-6, rtol=2e-6)

    def test_both_encoders_propagate_ranking_gradients(self):
        torch.manual_seed(42)
        for name in ["fingerprint", "graph"]:
            model = DirectRanker(name)
            fps = torch.rand(3, 2048)
            graphs = batch_graphs([graph(s) for s in ["CCC", "CCO", "CCN"]])
            logits = model(
                torch.randn(2, 768), fps, graphs, torch.tensor([[0, 1, 2], [1, 0, 2]])
            )
            loss = torch.nn.functional.cross_entropy(
                logits, torch.zeros(2, dtype=torch.long)
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(model.spectrum[1].weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.molecule_fp[0].weight.grad.abs().sum().item(), 0)
            if name == "graph":
                self.assertGreater(
                    model.graph.atom[0].weight.grad.abs().sum().item(), 0
                )

    def test_one_query_selection_is_repeatable_and_molecule_balanced(self):
        frame = pd.DataFrame(
            {"inchikey14": ["a", "a", "b", "b", "b", "c"], "row_id": range(6)}
        )
        selected = single_query(frame)
        self.assertEqual(selected.inchikey14.tolist(), ["a", "b", "c"])
        self.assertTrue(set(selected.row_id).issubset(frame.row_id))
        pd.testing.assert_frame_equal(selected, single_query(frame))


if __name__ == "__main__":
    unittest.main()
