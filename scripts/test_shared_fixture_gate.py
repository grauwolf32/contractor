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
    literal_paths,
    load_manifest,
    named_testdata_files,
    observed_reads,
    outside_problems,
    ui_reads,
)


def entry(*readers: str, reason: str = "") -> Entry:
    return Entry(frozenset(readers), reason)


class ManifestTest(unittest.TestCase):
    def test_repository_ui_imports_are_declared(self):
        manifest = load_manifest()
        imports = ui_reads()
        self.assertTrue(imports, "no UI shared fixture imports were checked")
        for fixture in imports:
            with self.subTest(fixture=fixture):
                declared = declared_readers(fixture, manifest)
                self.assertIsNotNone(declared)
                self.assertIn("ui", declared[1].readers)

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


class OutsideFixturesTest(unittest.TestCase):
    """A file that tests of two sides read is a shared fixture wherever it lives."""

    def tree(self, directory: str) -> Path:
        root = Path(directory)
        files = {
            "testdata/memory/cases.json": "{}",
            "internal/memory/testdata/local.json": "{}",
            "api/testdata/shared.json": "{}",
            "internal/memory/codec_test.go": "\n".join(
                [
                    "package memory",
                    'var a, _ = os.ReadFile("../../testdata/memory/cases.json")',
                    'var b, _ = os.ReadFile(filepath.Join("testdata", "local.json"))',
                    'var c = contractstest.ReadFile(t, "api", "testdata", "shared.json")',
                ]
            ),
            "runtime/tests/test_memory.py": "\n".join(
                [
                    'CASES = Path(__file__).parents[2] / "testdata" / "memory" / "cases.json"',
                    'SHARED = Path(__file__).parents[2] / "api/testdata/shared.json"',
                ]
            ),
            "ui/src/memory.test.ts": 'import cases from "../../testdata/memory/cases.json";',
        }
        for relative, text in files.items():
            (root / relative).parent.mkdir(parents=True, exist_ok=True)
            (root / relative).write_text(text)
        return root

    def test_literal_segments_join_into_one_path(self):
        self.assertEqual(
            literal_paths('filepath.Join("..", "..", "testdata", "x.json") + "y"'),
            ["../../testdata/x.json", "y"],
        )
        self.assertEqual(literal_paths('ROOT / "testdata" / "x.json"'), ["testdata/x.json"])
        self.assertEqual(literal_paths('read("a.json", name, "b.json")'), ["a.json", "b.json"])

    def test_two_sides_reading_a_file_outside_api_testdata_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            named = named_testdata_files(self.tree(directory))
        self.assertEqual(
            {fixture: set(sides) for fixture, sides in named.items()},
            {
                "testdata/memory/cases.json": {"go", "python", "ui"},
                "internal/memory/testdata/local.json": {"go"},
            },
        )
        self.assertEqual(
            outside_problems(named, {}),
            [
                "testdata/memory/cases.json: tests of 3 sides read it (go: "
                "internal/memory/codec_test.go; python: runtime/tests/test_memory.py; ui: "
                "ui/src/memory.test.ts); move it under api/testdata with a readers entry"
            ],
        )

    def test_a_named_outside_fixture_needs_a_live_entry(self):
        with tempfile.TemporaryDirectory() as directory:
            named = named_testdata_files(self.tree(directory))
        self.assertEqual(outside_problems(named, {"testdata/memory/": "reason"}), [])
        self.assertEqual(
            outside_problems(named, {"testdata/memory/": "reason", "testdata/gone.json": "reason"}),
            ["OUTSIDE_SHARED entry testdata/gone.json covers no file that two sides read"],
        )

    def test_repository_keeps_shared_fixtures_under_api_testdata(self):
        self.assertEqual(outside_problems(named_testdata_files()), [])


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


class CanonicalReportsTest(unittest.TestCase):
    def test_incomplete_canonical_reports_fail_without_rerunning_tests(self):
        import contextlib
        import io
        from unittest import mock
        import shared_fixture_gate as gate

        with tempfile.TemporaryDirectory() as directory:
            go = Path(directory) / "go.jsonl"
            log = Path(directory) / "reads.log"
            go.write_text("")
            log.write_text("")
            errors = io.StringIO()
            with (
                mock.patch.object(gate, "run", side_effect=AssertionError("report verification must not execute tests")),
                contextlib.redirect_stdout(io.StringIO()),
                contextlib.redirect_stderr(errors),
            ):
                self.assertEqual(gate.main(["--go-report", str(go), "--runtime-log", str(log)]), 1)
            self.assertIn("did not run", errors.getvalue())
            self.assertIn("does not read it", errors.getvalue())
