import unittest

from casmi_ml.metfrag_jfr_observation import aggregate_samples


def frame(owner, name):
    return {"method": {"type": {"name": owner}, "name": name}}


class MetfragJfrObservationTest(unittest.TestCase):
    def test_samples_count_recursive_membership_once_and_keep_empty_samples(self):
        rows = [
            {
                "type": "jdk.ExecutionSample",
                "values": {
                    "state": "STATE_RUNNABLE",
                    "stackTrace": {
                        "frames": [
                            frame("a/b", "match"),
                            frame("a/b", "match"),
                            frame("x/y", "caller"),
                        ]
                    },
                },
            },
            {
                "type": "jdk.ExecutionSample",
                "values": {"state": "STATE_RUNNABLE", "stackTrace": None},
            },
            {
                "type": "jdk.ObjectAllocationSample",
                "values": {"objectClass": {"name": "java/lang/Double"}, "weight": 4096},
            },
            {"type": "jdk.GarbageCollection", "values": {"duration": "PT0.012S"}},
        ]
        report = aggregate_samples(rows)
        assert report["execution_samples"] == 2
        assert report["sampled_thread_states"] == {"STATE_RUNNABLE": 2}
        assert dict(report["top_leaf_methods"]) == {"a.b.match": 1}
        assert dict(report["top_inclusive_methods"]) == {
            "a.b.match": 1,
            "x.y.caller": 1,
        }
        assert report["top_allocation_sample_weights_bytes"] == [
            ("java.lang.Double", 4096)
        ]
        assert report["gc_event_duration_sum_seconds"] == 0.012

    def test_unknown_duration_format_fails_without_inventing_cpu_time(self):
        with self.assertRaisesRegex(ValueError, "duration encoding"):
            aggregate_samples(
                [{"type": "jdk.GarbageCollection", "values": {"duration": "unknown"}}]
            )
