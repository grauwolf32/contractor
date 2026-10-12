import json
import tempfile
import unittest
from pathlib import Path

from shared_fixture_gate import (
    Entry,
    check_coverage,
    check_go_outcomes,
    check_python_outcomes,
    declared_readers,
    go_file_problems,
    go_outcomes,
    load_manifest,
    observed_reads,
)


def entry(*readers: str, reason: str = "") -> Entry:
    return Entry(frozenset(readers), reason)


class ManifestTest(unittest.TestCase):
    def test_the_longest_key_wins(self):
        manifest = {"v1alpha1/": entry("go", "python"), "v1alpha1/a.json": entry("go", "python", "ui")}
        self.assertEqual(declared_readers("v1alpha1/a.json", manifest)[0], "v1alpha1/a.json")
        self.assertEqual(declared_readers("v1alpha1/valid/b.json", manifest)[0], "v1alpha1/")
        self.assertIsNone(declared_readers("v1alpha2/b.json", manifest))
        # A file key never covers a longer path.
        self.assertIsNone(declared_readers("v1alpha1/a.json.bak", {"v1alpha1/a.json": entry("go")}))

    def test_manifest_rejects_unknown_sides_and_members(self):
        for readers in ({"readers": ["go", "java"]}, {"readers": []}, {"readers": ["go"], "why": "x"}):
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "readers.json"
                path.write_text(json.dumps({"description": "d", "readers": {"a.json": readers}}))
                with self.assertRaises(ValueError):
                    load_manifest(path)


class CoverageTest(unittest.TestCase):
    def test_every_rule_is_reported(self):
        manifest = {
            "both.json": entry("go", "python"),
            "lonely.json": entry("go"),
            "explained.json": entry("go", reason="only the Server reads it"),
            "unused.json": entry("go", "python"),
        }
        reads = {"both.json": {"go"}, "lonely.json": {"go", "ui"}, "explained.json": {"go"}}
        problems = check_coverage(["both.json", "lonely.json", "explained.json", "stray.json"], manifest, reads)
        for want in (
            "api/testdata/both.json: declared side python does not read it",
            "api/testdata/lonely.json: one reader (lonely.json) and no reason",
            "api/testdata/lonely.json: ui reads it but lonely.json does not declare ui",
            "api/testdata/stray.json: no entry in api/testdata/readers.json",
            "api/testdata/readers.json: entry unused.json covers no fixture",
        ):
            self.assertIn(want, problems)
        self.assertFalse(any("explained.json" in problem for problem in problems))


class GoSourcesTest(unittest.TestCase):
    def test_only_contractstest_calls_build_fixture_paths(self):
        text = "\n".join(
            [
                "package x",
                'var a = contractstest.ReadFile(t, "api", "testdata", "v1alpha1", "index.json")',
                'var b = filepath.Join(contractstest.Path(t, "api", "testdata", "v1alpha1"), "*.json")',
                '// os.ReadFile("../../api/testdata/v1alpha1/index.json") in a comment',
                'var c, _ = os.ReadFile("../../api/testdata/v1alpha1/index.json")',
                'var d = filepath.Join("..", "..", "api", "testdata", "x.json")',
            ]
        )
        self.assertEqual(
            go_file_problems("internal/x/x_test.go", text),
            [
                "internal/x/x_test.go:5: builds an api/testdata path; use contractstest",
                "internal/x/x_test.go:6: builds an api/testdata path; use contractstest",
            ],
        )

    def test_shared_tests_behind_build_constraints_are_reported(self):
        text = "//go:build integration\n\npackage x\n\nfunc TestSharedX(t *testing.T) {}\n"
        self.assertEqual(
            go_file_problems("internal/x/x_test.go", text),
            ["internal/x/x_test.go: TestShared tests behind '//go:build integration' do not run in the gate"],
        )


class OutcomesTest(unittest.TestCase):
    def test_go_tests_must_pass(self):
        events = [
            json.dumps({"Action": "pass", "Package": "m/internal/a", "Test": "TestSharedA"}),
            json.dumps({"Action": "pass", "Package": "m/internal/a", "Test": "TestSharedA/case"}),
            json.dumps({"Action": "skip", "Package": "m/internal/b", "Test": "TestSharedB"}),
            json.dumps({"Action": "fail", "Package": "m/internal/c", "Test": "TestSharedC"}),
            "not json",
        ]
        outcomes = go_outcomes(events, "m")
        self.assertEqual(outcomes[("internal/a", "TestSharedA")], "pass")
        discovered = {
            ("internal/a", "TestSharedA"): "a_test.go",
            ("internal/b", "TestSharedB"): "b_test.go",
            ("internal/c", "TestSharedC"): "c_test.go",
            ("internal/d", "TestSharedD"): "d_test.go",
        }
        self.assertEqual(
            check_go_outcomes(discovered, outcomes),
            [
                "Go: TestSharedB (b_test.go) skip",
                "Go: TestSharedC (c_test.go) fail",
                "Go: TestSharedD (d_test.go) did not run",
            ],
        )

    def test_zero_tests_fail(self):
        self.assertIn("Go: no TestShared* test exists", check_go_outcomes({}, {}))
        self.assertIn("Go: the shared test run passed no test", check_go_outcomes({}, {}))
        self.assertIn("Python: no test_shared_* test exists", check_python_outcomes({}, []))

    def test_python_tests_must_pass_every_case(self):
        records = [
            "python-test\ttests/test_a.py::test_shared_a[1]\tcall:passed",
            "python-test\ttests/test_a.py::test_shared_a[2]\tcall:failed",
            "python-test\ttests/test_b.py::test_shared_b\tsetup:skipped",
            "python\ttests/test_a.py::test_shared_a[1]\tapi/testdata/x.json",
            "go\tinternal/a\tTestSharedA\tapi/testdata/x.json",
        ]
        discovered = {"test_a.py::test_shared_a": "a", "test_b.py::test_shared_b": "b"}
        self.assertEqual(
            check_python_outcomes(discovered, records),
            [
                "Python: tests/test_a.py::test_shared_a[2] call:failed",
                "Python: tests/test_b.py::test_shared_b setup:skipped",
                "Python: test_b.py::test_shared_b (b) did not run and pass",
            ],
        )


class ReadsTest(unittest.TestCase):
    def test_reads_count_only_for_passing_go_tests_and_shared_modules(self):
        records = [
            "go\tinternal/a\tTestSharedA\tapi/testdata/a.json",
            "go\tinternal/b\tTestSharedB\tapi/testdata/b.json",
            "python\ttests/test_a.py::test_shared_a\tapi/testdata/a.json",
            "python-module\truntime/tests/test_c.py\tapi/testdata/c.json",
            "python-module\truntime/tests/fakes/spec.py\tapi/testdata/d.json",
        ]
        reads = observed_reads(records, {("internal/a", "TestSharedA")}, {"test_c.py"})
        self.assertEqual(reads, {"a.json": {"go", "python"}, "c.json": {"python"}})


if __name__ == "__main__":
    unittest.main()
