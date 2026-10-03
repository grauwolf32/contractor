"""Every scan Worker errorCode is registered in the table the Planner and UI share."""

import ast
import json
from pathlib import Path

import contractor_runtime.toolsets.scan as scan_package

ROOT = Path(__file__).parents[2]
CASES = json.loads((ROOT / "api/scan/v1/testdata/worker-error-codes.json").read_text())
CODE_TARGETS = {"error", "error_code", "errorCode"}
CODE_EXCEPTIONS = {"ScannerUnavailable", "ScanTargetRefused"}


def _string_constants(node: ast.AST) -> set[str]:
    # Codes are literal operands; subscript keys and call arguments are not codes.
    if isinstance(node, ast.Constant):
        return {node.value} if isinstance(node.value, str) else set()
    if isinstance(node, ast.IfExp):
        return _string_constants(node.body) | _string_constants(node.orelse)
    if isinstance(node, ast.BoolOp):
        return set().union(*(_string_constants(value) for value in node.values))
    return set()


def _emitted_codes() -> set[str]:
    codes: set[str] = set()
    for path in Path(scan_package.__file__).parent.glob("*.py"):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            keyword = isinstance(node, ast.keyword) and node.arg in CODE_TARGETS
            assignment = isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id in CODE_TARGETS
                for target in node.targets
            )
            if keyword or assignment:
                codes |= _string_constants(node.value)
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in CODE_EXCEPTIONS
            ):
                codes |= _string_constants(node.args[0])
    return codes


def test_every_emitted_scan_error_code_has_a_shared_outcome() -> None:
    registered = {case["errorCode"] for case in CASES["codes"]}
    assert len(registered) == len(CASES["codes"])
    assert _emitted_codes() == registered
    assert {case["outcome"] for case in CASES["codes"]} <= set(CASES["outcomes"])
