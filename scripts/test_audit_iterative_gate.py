"""False-green, provenance and prerequisite proofs for the iterative release gate."""
import contextlib
import importlib.util
import io
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

spec = importlib.util.spec_from_file_location("iterative_gate", Path(__file__).with_name("test-audit-iterative.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
PACKAGE = "github.com/grauwolf32/contractor/tests/e2e"


def selection():
    cases = {PACKAGE: ["TestJourney/required"]}
    harness = {"test_audit_iterative_gate": ["AuditIterativeGateTest.test_missing_prerequisites_fail_before_commands"]}
    return {"postgres": cases, "process": cases, "harness": harness}


def manifest():
    return {"schema_version": "2.0", "cases": [
        {"boundary": boundary, "package": package, "case": case,
         "covers": [f"A{i}" for i in range(1, 7)] if boundary == "postgres" else []}
        for boundary, packages in selection().items()
        for package, cases in packages.items() for case in cases
    ]}


def events():
    return [dict(Action=action, Package=PACKAGE, Test=case) for action, case in (
        ("run", "TestJourney"), ("run", "TestJourney/required"),
        ("pass", "TestJourney/required"), ("pass", "TestJourney"),
    )] + [dict(Action="pass", Package=PACKAGE)]


def write_report(path, values):
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


class AuditIterativeGateTest(unittest.TestCase):
    def test_zero_missing_skipped_failed_and_out_of_order_reports_fail(self):
        complete = events()
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate, "matrix", return_value=selection()):
            report = Path(temporary) / "report.jsonl"
            write_report(report, complete + [dict(Action="skip", Package=PACKAGE, Test="TestOptional"), dict(Action="skip", Package="optional", Test="TestJourney")])
            self.assertEqual(gate.verify(report, "postgres"), [PACKAGE + "/TestJourney", PACKAGE + "/TestJourney/required"])
            vectors = [[], complete[:1] + complete[3:], complete[:-1], complete[1:],
                       [complete[2], *complete[:2], *complete[3:]], complete + [complete[0]],
                       complete + [dict(Action="skip", Package=PACKAGE, Test="TestJourney/required")],
                       complete + [dict(Action="fail", Package=PACKAGE)],
                       complete + [dict(Action="fail", Package=PACKAGE, Test="TestJourney/future_fault")],
                       [*complete[:2], dict(Action="run", Package=PACKAGE, Test="TestJourney/unlisted_unfinished"), *complete[2:]],
                       complete + [dict(Action="run", Package=PACKAGE, Test="TestJourney/late")],
                       complete + [dict(Action="output", Package=PACKAGE, Test="TestJourney/required", Output="late")]]
            completed_extra = [*complete[:2], dict(Action="run", Package=PACKAGE, Test="TestJourney/unlisted_complete"), dict(Action="pass", Package=PACKAGE, Test="TestJourney/unlisted_complete"), *complete[2:]]
            write_report(report, completed_extra)
            self.assertEqual(gate.verify(report, "postgres"), [PACKAGE + "/TestJourney", PACKAGE + "/TestJourney/required", PACKAGE + "/TestJourney/unlisted_complete"])
            vectors += [complete + [dict(Action="build-fail", Package=PACKAGE)], complete + [complete[-1]], complete[:2] + [complete[1]] + complete[2:],
                        [complete[1], complete[0], *complete[2:]],
                        [complete[0], complete[3], complete[1], complete[2], complete[-1]],
                        [*complete[:2], complete[3], complete[2], complete[-1]],
                        [*complete[:2], dict(Action="run", Package=PACKAGE, Test="TestJourney/unlisted"), complete[2], complete[3], dict(Action="pass", Package=PACKAGE, Test="TestJourney/unlisted"), complete[-1]]]
            for vector in vectors:
                write_report(report, vector)
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    gate.verify(report, "postgres")
            parallel = selection()
            other = PACKAGE + "/other"
            parallel["postgres"] = {PACKAGE: ["TestJourney/required"], other: ["TestJourney/required"]}
            interleaved = [*complete[:2], *(dict(event, Package=other) for event in complete[:2]), *complete[2:], *(dict(event, Package=other) for event in complete[2:])]
            write_report(report, interleaved)
            with mock.patch.object(gate, "matrix", return_value=parallel):
                self.assertEqual(len(gate.verify(report, "postgres")), 4)
            for malformed in ('not json\n', '[]\n', '{"Action":"run","Action":"pass"}\n', '{"Action":1}\n', 'x' * 65537):
                report.write_text(malformed)
                with self.subTest(malformed=malformed[:40]), self.assertRaises(ValueError):
                    gate.verify(report, "postgres")

    def test_matrix_rejects_empty_stale_and_unmapped_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.yml"
            with mock.patch.object(gate, "MATRIX", path):
                path.write_text(json.dumps(manifest()))
                self.assertEqual(gate.matrix(), selection())
                for boundary in ("postgres", "process", "harness"):
                    bad = manifest()
                    bad["cases"] = [case for case in bad["cases"] if case["boundary"] != boundary]
                    path.write_text(json.dumps(bad))
                    with self.subTest(boundary=boundary), self.assertRaises(ValueError):
                        gate.matrix()
                for mutate in (
                    lambda value: value.update(cases=[]),
                    lambda value: value["cases"].append(value["cases"][0]),
                    lambda value: value["cases"][0].update(covers=[]),
                    lambda value: value["cases"][0].update(covers=["A7"]),
                    lambda value: value["cases"][0].update(covers=["A1", "A1"]),
                    lambda value: value["cases"][0].update(case="not a Go case"),
                    lambda value: value["cases"][-1].update(case="AuditIterativeGateTest.test_absent"),
                ):
                    bad = manifest()
                    mutate(bad)
                    path.write_text(json.dumps(bad))
                    with self.subTest(bad=bad), self.assertRaises(ValueError):
                        gate.matrix()
                # Supporting regressions have no acceptance link but remain selected.
                value = manifest()
                value["cases"].append(dict(value["cases"][0], case="TestSupporting", covers=[]))
                path.write_text(json.dumps(value))
                self.assertIn("TestSupporting", gate.matrix()["postgres"][PACKAGE])
        with self.assertRaises(ValueError):
            gate.pattern([])
        self.assertEqual(gate.pattern(["TestA/sub", "TestB", "TestA/other"]), "^(TestA|TestB)$")

    def test_missing_prerequisites_fail_before_commands(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate.reports, "capture") as capture:
            with mock.patch.dict(os.environ, {"CONTRACTOR_TEST_DATABASE_URL": ""}), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "DATABASE_URL"):
                    gate.run("postgres", Path(temporary))
            capture.assert_not_called()
            result = json.loads(next(Path(temporary).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "failed")
            self.assertRegex(result["source"]["revision"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(result["source"]["dirty"], bool)
            for boundary in ("postgres", "process"):
                with mock.patch.dict(os.environ, {"CONTRACTOR_TEST_DATABASE_URL": "configured"}), mock.patch.object(gate.shutil, "which", return_value=None):
                    with self.assertRaisesRegex(ValueError, "Go"):
                        gate._prerequisites(boundary)
            with mock.patch.dict(os.environ, {"CONTRACTOR_TEST_DATABASE_URL": "configured"}), mock.patch.object(gate.shutil, "which", side_effect=lambda tool: "go" if tool == "go" else None):
                with self.assertRaisesRegex(ValueError, "Runtime"):
                    gate._prerequisites("process")

    def test_stale_evidence_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate, "matrix", return_value=selection()), mock.patch.object(gate, "_prerequisites"), mock.patch.object(gate.reports, "source_state", return_value={"revision": "a" * 40, "dirty": False}):
            output = Path(temporary)
            old = output / "postgres-old"
            old.mkdir()
            (old / "result.json").write_text('{"status":"passed"}')
            def no_tests(report, _command, **_options):
                write_report(report, [dict(Action="pass", Package=PACKAGE)])
                return 0
            with mock.patch.object(gate.reports, "capture", side_effect=no_tests), mock.patch.object(gate, "MATRIX", Path(__file__)), contextlib.redirect_stdout(io.StringIO()):
                with self.assertRaisesRegex(ValueError, "complete"):
                    gate.run("postgres", output)
            fresh = [path for path in output.glob("*/result.json") if path.parent != old]
            self.assertEqual(len(fresh), 1)
            self.assertEqual(json.loads(fresh[0].read_text())["status"], "failed")
            self.assertEqual(json.loads((old / "result.json").read_text())["status"], "passed")

    def test_changed_source_cannot_publish_passing_evidence(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate, "matrix", return_value=selection()), mock.patch.object(gate, "_prerequisites"), mock.patch.object(gate, "MATRIX", Path(__file__)), contextlib.redirect_stdout(io.StringIO()):
            def completed(report, _command, **_options):
                write_report(report, events())
                return 0
            before = {"revision": "a" * 40, "dirty": False}
            after = {"revision": "b" * 40, "dirty": False}
            with mock.patch.object(gate.reports, "source_state", side_effect=[before, before, after]), mock.patch.object(gate.reports, "capture", side_effect=completed), self.assertRaisesRegex(ValueError, "source"):
                gate.run("postgres", Path(temporary))
            result = json.loads(next(Path(temporary).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "failed")

    def test_dirty_source_cannot_publish_execution_evidence(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate, "_prerequisites"), mock.patch.object(gate.reports, "source_state", return_value={"revision": "a" * 40, "dirty": True}), mock.patch.object(gate.reports, "capture") as capture, contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaisesRegex(ValueError, "clean source"):
                gate.run("postgres", Path(temporary))
            capture.assert_not_called()
            result = json.loads(next(Path(temporary).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "failed")

    def test_changed_matrix_cannot_publish_execution_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.yml"
            path.write_text(json.dumps(manifest()))
            def completed(report, _command, **_options):
                write_report(report, events())
                path.write_text(path.read_text() + " ")
                return 0
            with mock.patch.object(gate, "MATRIX", path), mock.patch.object(gate, "_prerequisites"), mock.patch.object(gate.reports, "source_state", return_value={"revision": "a" * 40, "dirty": False}), mock.patch.object(gate.reports, "capture", side_effect=completed), contextlib.redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "matrix"):
                gate.run("postgres", Path(temporary) / "evidence")

    def test_bounded_redacted_evidence_records_source_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / "go"
            payload = events()
            executable.write_text(f"#!{sys.executable}\nimport json,os\nprint(json.dumps({{'Action':'output','Package':{PACKAGE!r},'Output':os.environ['CONTRACTOR_TEST_DATABASE_URL']+' fixture-target-body-marker'}}))\nfor event in {payload!r}: print(json.dumps(event))\n")
            executable.chmod(0o700)
            manifest_path = directory / "matrix.yml"
            manifest_path.write_text(json.dumps(manifest()))
            env = {"PATH": str(directory) + os.pathsep + os.environ.get("PATH", ""), "CONTRACTOR_TEST_DATABASE_URL": "postgres://u:secret-value@localhost/db"}
            with mock.patch.dict(os.environ, env), mock.patch.object(gate, "MATRIX", manifest_path), mock.patch.object(gate.reports, "source_state", return_value={"revision": "a" * 40, "dirty": False}), contextlib.redirect_stdout(io.StringIO()) as output:
                result = gate.run("postgres", directory / "evidence")
            document = json.loads(result.read_text())
            self.assertEqual(document["status"], "passed")
            self.assertRegex(document["source"]["revision"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(document["source"]["dirty"], bool)
            self.assertEqual(document["counts"], {"required": 1, "passed": 2, "skipped": 0})
            self.assertNotIn("coverage", document)
            self.assertEqual(document["matrix"]["path"], "matrix.yml")
            self.assertEqual(document["cases"], [PACKAGE + "/TestJourney", PACKAGE + "/TestJourney/required"])
            self.assertLessEqual(result.stat().st_size, gate.MAX_EVIDENCE_BYTES)
            retained = result.read_text() + output.getvalue() + (result.parent / "go.jsonl").read_text()
            self.assertNotIn("secret-value", retained)
            self.assertNotIn("fixture-target-body-marker", retained)

    def test_process_failure_and_cancellation_record_incomplete_verification(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate, "_prerequisites"), mock.patch.object(gate.reports, "source_state", return_value={"revision": "a" * 40, "dirty": False}), mock.patch.object(gate, "matrix", return_value=selection()), mock.patch.object(gate, "MATRIX", Path(__file__)), contextlib.redirect_stdout(io.StringIO()):
            for error in (KeyboardInterrupt(), TimeoutError(), ValueError()):
                with mock.patch.object(gate.reports, "capture", side_effect=error), self.assertRaises(type(error)):
                    gate.run("process", Path(temporary))
            statuses = {json.loads(path.read_text())["status"] for path in Path(temporary).glob("*/result.json")}
            self.assertEqual(statuses, {"cancelled", "timed_out", "failed"})


if __name__ == "__main__":
    unittest.main()
