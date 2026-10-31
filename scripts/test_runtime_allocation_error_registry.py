"""Extend ERR-R24's failure-code gate to the Runtime's allocation boundary."""

import ast
from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parent.parent
OWNER = re.compile(r"\[[A-Z]+-R[0-9]+\]\(")


def allocation_codes(source, filename):
    codes = {}
    for node in ast.walk(ast.parse(source, filename=filename)):
        if not isinstance(node, ast.Call):
            continue
        name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", None)
        if name != "AllocationError":
            continue
        code = (
            node.args[0]
            if node.args
            else next(keyword.value for keyword in node.keywords if keyword.arg == "code")
        )
        # Domain errors keep their existing code when preparation wraps them.
        # Their vocabularies belong to their domain gates, not this boundary.
        if ast.unparse(code) in {"error.code", "error.code.value"}:
            continue
        if not isinstance(code, ast.Constant) or not isinstance(code.value, str):
            raise ValueError(f"{filename}:{node.lineno}: unresolved AllocationError code")
        codes.setdefault(code.value, []).append(f"{filename}:{node.lineno}")
    return codes


def registry_errors(codes, registry):
    _, separator, section = registry.partition("\n## Registry\n")
    if not separator:
        return ["errors.md has no Registry section"]
    rows = {}
    for line in section.splitlines():
        match = re.match(r"^\| `([a-z][a-z0-9_]*)` \|", line)
        if match:
            owner = line.strip("| ").split(" | ")[-1]
            rows.setdefault(match[1], []).append(owner)
    errors = []
    for code, positions in sorted(codes.items()):
        owners = rows.get(code, [])
        if len(owners) != 1 or not OWNER.search(owners[0]):
            errors.append(f"{', '.join(positions)}: {code} needs exactly one owned registry row")
    return errors


class RuntimeAllocationErrorRegistryTest(unittest.TestCase):
    def test_runtime_allocation_codes_are_registered(self):
        codes = {}
        for path in sorted((ROOT / "runtime/src/contractor_runtime").rglob("*.py")):
            for code, positions in allocation_codes(
                path.read_text(), str(path.relative_to(ROOT))
            ).items():
                codes.setdefault(code, []).extend(positions)
        # Pin each emitting module, including the lifecycle HTTP handler and
        # the helper functions that construct rather than raise the error.
        for code in (
            "invalid_toolset_factory",
            "unsupported_toolset",
            "allocation_conflict",
            "allocation_id_mismatch",
        ):
            self.assertIn(code, codes)
        registry = (ROOT / "docs/spec/platform/errors.md").read_text()
        self.assertEqual(registry_errors(codes, registry), [])

    def test_gate_rejects_an_unregistered_code(self):
        source = """
raise AllocationError(
    "unregistered_allocation_code", "safe message", retryable=False, status_code=422,
)
"""
        codes = allocation_codes(source, "probe.py")
        self.assertEqual(codes, {"unregistered_allocation_code": ["probe.py:2"]})
        registry = (ROOT / "docs/spec/platform/errors.md").read_text()
        self.assertEqual(len(registry_errors(codes, registry)), 1)
        self.assertIn("unregistered_allocation_code", registry_errors(codes, registry)[0])

    def test_registry_row_needs_one_requirement_owner(self):
        codes = {"probe": ["probe.py:1"]}
        row = "| `probe` | — | no | Safe error | [ERR-R24](errors.md#ERR-R24) |\n"
        self.assertEqual(registry_errors(codes, "\n## Registry\n" + row), [])
        self.assertTrue(registry_errors(codes, "\n## Registry\n" + row + row))
        self.assertTrue(
            registry_errors(
                codes, "\n## Registry\n" + row.replace("[ERR-R24](errors.md#ERR-R24)", "unknown")
            )
        )

    def test_named_codes_and_domain_wrapping(self):
        source = """
AllocationError(code="named_code", message="safe", retryable=False, status_code=422)
AllocationError(error.code, "safe", retryable=error.retryable, status_code=422)
AllocationError(error.code.value, "safe", retryable=False, status_code=422)
"""
        self.assertEqual(allocation_codes(source, "probe.py"), {"named_code": ["probe.py:2"]})
        with self.assertRaisesRegex(ValueError, "unresolved AllocationError code"):
            allocation_codes('AllocationError(hidden_code, "safe")', "probe.py")


if __name__ == "__main__":
    unittest.main()
