import hashlib
import struct
import unittest

from casmi_ml.metfrag_polling_patch import replace_wait_constant


class PollingConstantPatchTests(unittest.TestCase):
    def sample(self, waits):
        return (
            b"\xca\xfe\xba\xbe"
            + struct.pack(">HHH", 0, 65, 1 + len(waits) * 2)
            + b"".join(b"\x05" + struct.pack(">q", n) for n in waits)
            + b"unchanged-methods"
        )

    def test_only_single_wait_constant_changes_with_same_class_size(self):
        data = self.sample([1000, 20])
        result = replace_wait_constant(data, hashlib.sha256(data).hexdigest())
        self.assertEqual(result, self.sample([10, 20]))
        self.assertEqual(len(data), len(result))

    def test_wrong_binding_absent_or_duplicate_wait_are_rejected(self):
        data = self.sample([1000])
        with self.assertRaisesRegex(ValueError, "Pinned"):
            replace_wait_constant(data, "wrong")
        for waits in ([20], [1000, 1000]):
            data = self.sample(waits)
            with self.assertRaisesRegex(ValueError, "Exactly one"):
                replace_wait_constant(data, hashlib.sha256(data).hexdigest())
