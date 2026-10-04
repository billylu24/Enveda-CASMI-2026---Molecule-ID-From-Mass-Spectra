import unittest

from casmi_ml.chemistry import composition_mass
from casmi_ml.generation_recall_funnel_audit import inside_mass


class GenerationRecallFunnelTest(unittest.TestCase):
    def test_original_absolute_and_ppm_tolerance_and_empty_window(self):
        mass = composition_mass("C2H6O")
        self.assertTrue(inside_mass("C2H6O", [mass + 0.005]))
        self.assertFalse(inside_mass("C2H6O", [mass + 0.007]))
        self.assertFalse(inside_mass("C2H6O", []))
        large = composition_mass("C100H202")
        self.assertTrue(inside_mass("C100H202", [large + 0.02]))
        self.assertFalse(inside_mass("C100H202", [large + 0.08]))
