import unittest
from pathlib import Path
from unittest.mock import patch

from casmi_ml.chembl_fragment_pilot import main


class ChemblFragmentCLITests(unittest.TestCase):
    def test_wide_pool_parameters_reach_actual_run(self):
        argv = [
            "pilot",
            "--output",
            "out",
            "--incumbent",
            "best",
            "--limit",
            "200",
            "--prefix",
            "10",
            "--proposal-limit",
            "500",
            "--fragment-limit",
            "25",
            "--monomer",
        ]
        with (
            patch("sys.argv", argv),
            patch("casmi_ml.chembl_fragment_pilot.run", return_value={}) as run,
            patch("builtins.print"),
        ):
            main()
        run.assert_called_once_with(
            Path("out"), Path("best"), 200, 10, True, 500, 25, False
        )
