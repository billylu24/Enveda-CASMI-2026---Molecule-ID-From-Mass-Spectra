"""Emit completed original per-molecule handoffs without changing generator statements."""

import ast
import inspect
import json
import os
from pathlib import Path

from casmi_ml import generation_inference as original
from casmi_ml.data import write_json
from casmi_ml.metfrag import digest


class HandoffEmitter(ast.NodeTransformer):
    def visit_Expr(self, node):
        self.generic_visit(node)
        call = node.value
        if (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name)
            and call.func.value.id == "final_full"
            and call.func.attr == "append"
        ):
            emit = ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="_casmi_emit_completed_handoff", ctx=ast.Load()),
                    args=[
                        ast.Subscript(
                            value=ast.Name(id="final_full", ctx=ast.Load()),
                            slice=ast.Constant(-1),
                            ctx=ast.Load(),
                        )
                    ],
                    keywords=[],
                )
            )
            return [node, ast.copy_location(emit, node)]
        return node


def build_predict(emit):
    """Original AST plus callbacks immediately after each completed handoff append."""
    source = inspect.getsource(original.predict)
    tree = HandoffEmitter().visit(ast.parse(source))
    callbacks = sum(
        isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "_casmi_emit_completed_handoff"
        for n in ast.walk(tree)
    )
    if callbacks != 2:
        raise ValueError("Original normal and timeout handoff append sites required")
    namespace = dict(vars(original), _casmi_emit_completed_handoff=emit)
    # Only the inspected repository function is compiled; no external code input.
    exec(compile(ast.fix_missing_locations(tree), original.__file__, "exec"), namespace)  # noqa: S102
    return namespace["predict"]


def emit_to_directory(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    sequence = 0

    def emit(row):
        nonlocal sequence
        if set(row) != {"molecule_id", "smiles"} or not isinstance(row["smiles"], list):
            raise ValueError("Original complete generated handoff required")
        target = directory / f"{sequence:06d}.json"
        temporary = directory / f"{sequence:06d}.pending"
        write_json(temporary, row)
        os.replace(temporary, target)
        sequence += 1

    return emit


def run_worker(config):
    import time

    payload = json.loads(Path(config).read_text())
    remaining = payload["deadline"] - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("Original shared generation deadline expired")
    arguments = payload["arguments"]
    arguments["seconds"] = remaining
    folder = Path(config).parent / "handoff_events"
    predict = build_predict(emit_to_directory(folder))
    predict(**arguments)
    write_json(
        folder / "complete.json",
        {
            "worker_output": arguments["output"],
            "source_sha256": digest(__file__),
            "original_generation_source_sha256": digest(original.__file__),
            "original_deadline_retained": True,
        },
    )


def main():
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--worker", required=True)
    args = p.parse_args()
    run_worker(args.worker)


if __name__ == "__main__":
    main()
