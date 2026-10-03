import unittest

import numpy as np
import pandas as pd

from casmi_ml.dino_energy_pairs import choose_pairs, pairing_table


class EnergyPairTests(unittest.TestCase):
    def test_same_molecule_ion_instrument_and_distinct_valid_energy_only(self):
        frame = pd.DataFrame(
            {
                "inchikey14": ["a", "a", "a", "a", "a", "b"],
                "adduct": ["H", "H", "Na", "H", "H", "H"],
                "instrument_type": [
                    "Orbitrap",
                    "Orbitrap",
                    "Orbitrap",
                    "QTOF",
                    "Orbitrap",
                    "Orbitrap",
                ],
                "collision_energy_ev": [[20], [40], [40], [40], [float("nan")], [40]],
            }
        )
        table = pairing_table(frame)
        self.assertEqual([x.tolist() for x in table], [[1], [0], [], [], [], []])
        first, second = choose_pairs(
            frame.groupby("inchikey14", sort=True).indices,
            table,
            np.random.default_rng(42),
        )
        for a, b in zip(first, second):
            if a != b:
                self.assertIn(b, table[a])
            self.assertEqual(frame.iloc[a].inchikey14, frame.iloc[b].inchikey14)
