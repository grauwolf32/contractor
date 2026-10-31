"""Keep executable Runtime bounds on the fixed contracts.base definition."""

import ast
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parent.parent
CANONICAL = "runtime/src/contractor_runtime/contracts/base.py"
BOUNDARY_VALUES = {9007199254740991, 9007199254740992}


def numeric_literal(node):
    if isinstance(node, ast.Constant):
        if type(node.value) in (int, float):
            return node.value
        if isinstance(node.value, str) and node.value.lstrip("+-").isdigit():
            return int(node.value)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = numeric_literal(node.operand)
        return None if value is None else (-value if isinstance(node.op, ast.USub) else value)
    if isinstance(node, ast.BinOp):
        left, right = numeric_literal(node.left), numeric_literal(node.right)
        if left is None or right is None:
            return None
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Add):
            return left + right
        if type(left) is int and type(right) is int and 0 <= right <= 64:
            if isinstance(node.op, ast.LShift):
                return left << right
            if isinstance(node.op, ast.Pow) and abs(left) <= 64:
                return left**right
    return None


def copied_bounds(source, name):
    tree = ast.parse(source, filename=name)
    canonical = {
        id(node.value)
        for node in ast.walk(tree)
        if name == CANONICAL
        and isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "MAX_EXACT_INTEGER"
            for target in node.targets
        )
    }
    found = []

    def visit(node):
        if id(node) in canonical:
            return
        if (
            isinstance(node, ast.Expr)
            and isinstance(node.value, ast.Constant)
            and isinstance(node.value.value, str)
        ):
            return  # Documentation/examples are not executable bound definitions.
        value = numeric_literal(node)
        if value is not None and abs(value) in BOUNDARY_VALUES:
            found.append(f"{name}:{node.lineno}")
            return
        for child in ast.iter_child_nodes(node):
            visit(child)

    visit(tree)
    return found


class ExactIntegerDefinitionTest(unittest.TestCase):
    def test_runtime_has_one_numeric_definition(self):
        files = sorted((ROOT / "runtime/src").rglob("*.py"))
        self.assertGreater(len(files), 100)
        copies = [
            position
            for path in files
            for position in copied_bounds(path.read_text(), str(path.relative_to(ROOT)))
        ]
        self.assertEqual(copies, [])

    def test_guard_rejects_literal_and_computed_copies(self):
        for expression in (
            "9007199254740991",
            "2**53 - 1",
            "(1 << 53) - 1",
            "2**53",
            '"9007199254740991"',
        ):
            with self.subTest(expression=expression):
                self.assertEqual(len(copied_bounds("value = " + expression, "probe.py")), 1)
        self.assertEqual(copied_bounds("value = MAX_EXACT_INTEGER", "probe.py"), [])
        self.assertEqual(copied_bounds("MAX_EXACT_INTEGER = 2**53 - 1", CANONICAL), [])
        self.assertEqual(
            len(copied_bounds("MAX_EXACT_INTEGER = 2**53 - 1\nother = 2**53 - 1", CANONICAL)), 1
        )


if __name__ == "__main__":
    unittest.main()
