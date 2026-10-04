import copy
import unittest

from casmi_ml.one_time_kaggle import verify_runtime_bindings


class RuntimeSourceBindingTest(unittest.TestCase):
    def test_cpu_worker_source_and_count_must_match(self):
        local = {
            "generation_cpu_runtime": {
                "workers": 2,
                "torch_cpu_threads": 1,
                "worker_source_sha256": "bound",
                "pipeline_source_sha256": "bound",
            }
        }
        verify_runtime_bindings(local, copy.deepcopy(local))
        for key, value in (
            ("workers", 0),
            ("torch_cpu_threads", 4),
            ("worker_source_sha256", "changed"),
            ("pipeline_source_sha256", "changed"),
        ):
            with self.subTest(key=key):
                platform = copy.deepcopy(local)
                platform["generation_cpu_runtime"][key] = value
                with self.assertRaisesRegex(ValueError, "runtime/source differs"):
                    verify_runtime_bindings(local, platform)
        with self.assertRaises(ValueError):
            verify_runtime_bindings(local, {})

    def test_overlap_timings_can_differ_but_all_events_and_sources_must_match(self):
        local = {
            "generation_physical_overlap": {
                "emitted_molecules": 400,
                "stream_source_sha256": "a",
                "overlap_source_sha256": "b",
                "pipeline_source_sha256": "c",
                "generation_and_early_planning_seconds": 100,
            }
        }
        platform = copy.deepcopy(local)
        platform["generation_physical_overlap"][
            "generation_and_early_planning_seconds"
        ] = 200
        verify_runtime_bindings(local, platform)
        platform["generation_physical_overlap"]["emitted_molecules"] = 399
        with self.assertRaisesRegex(ValueError, "source/events differ"):
            verify_runtime_bindings(local, platform)

    def test_mass_reuse_classes_sources_and_deadlines_are_bound(self):
        local = {
            "mass_reuse_runtime": {
                "classes": {"fragment": "sha"},
                "sources": {"pipeline": "sha"},
                "original_deadlines_retained": True,
            }
        }
        verify_runtime_bindings(local, copy.deepcopy(local))
        for key, changed in (
            ("classes", {"fragment": "other"}),
            ("sources", {"pipeline": "other"}),
            ("original_deadlines_retained", False),
        ):
            with self.subTest(key=key):
                platform = copy.deepcopy(local)
                platform["mass_reuse_runtime"][key] = changed
                with self.assertRaisesRegex(ValueError, "mass-reuse"):
                    verify_runtime_bindings(local, platform)
        with self.assertRaises(ValueError):
            verify_runtime_bindings(local, {})

    def test_older_runtime_still_uses_original_gates(self):
        verify_runtime_bindings({}, {})
