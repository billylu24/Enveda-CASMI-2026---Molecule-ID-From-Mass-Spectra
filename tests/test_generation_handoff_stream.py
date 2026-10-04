import ast
import inspect
import json
import tempfile
import unittest
from pathlib import Path

from casmi_ml import generation_inference as original
from casmi_ml.generation_handoff_stream import (
    HandoffEmitter,
    build_predict,
    emit_to_directory,
)


class HandoffStreamTest(unittest.TestCase):
    def test_only_new_statements_are_handoff_callbacks(self):
        source = inspect.getsource(original.predict)
        tree = HandoffEmitter().visit(ast.parse(source))
        count = 0

        class StripCallbacks(ast.NodeTransformer):
            def visit_Expr(self, node):
                nonlocal count
                if (
                    isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == "_casmi_emit_completed_handoff"
                ):
                    count += 1
                    return None
                return self.generic_visit(node)

        stripped = StripCallbacks().visit(tree)
        self.assertEqual(count, 2)
        self.assertEqual(ast.dump(stripped), ast.dump(ast.parse(source)))
        self.assertEqual(
            inspect.signature(build_predict(lambda row: None)),
            inspect.signature(original.predict),
        )

    def test_atomic_event_keeps_complete_original_smiles_and_opaque_identity(self):
        with tempfile.TemporaryDirectory() as root:
            emit = emit_to_directory(root)
            rows = [
                {"molecule_id": "opaque/query", "smiles": ["CCO", "COC"]},
                {"molecule_id": "another", "smiles": ["CCC"]},
            ]
            for row in rows:
                emit(row)
            paths = sorted(Path(root).glob("*.json"))
            self.assertEqual([json.loads(p.read_text()) for p in paths], rows)
            self.assertFalse(list(Path(root).glob("*.pending")))
            self.assertEqual([p.name for p in paths], ["000000.json", "000001.json"])


if __name__ == "__main__":
    unittest.main()
