"""False-green, provenance and prerequisite proofs for the iterative release gate."""
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

spec = importlib.util.spec_from_file_location("iterative_gate", Path(__file__).with_name("test-audit-iterative.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)
PACKAGE = "github.com/grauwolf32/contractor/tests/e2e"


def selection():
    cases = {PACKAGE: ["TestJourney/required"]}
    harness = {"test_audit_iterative_gate": ["AuditIterativeGateTest.test_missing_prerequisites_fail_before_commands"]}
    return {"schema_version": "1.0", "postgres": cases, "process": cases, "harness": harness,
            "coverage": {f"A{i}": [{"boundary": "postgres", "package": PACKAGE, "case": "TestJourney/required"}] for i in range(1, 7)}}


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
            self.assertEqual(gate.verify(report, "postgres"), [PACKAGE + "/TestJourney/required"])
            vectors = [[], complete[:1] + complete[3:], complete[:-1], complete[1:],
                       [complete[2], *complete[:2], *complete[3:]], complete + [complete[0]],
                       complete + [dict(Action="skip", Package=PACKAGE, Test="TestJourney/required")],
                       complete + [dict(Action="fail", Package=PACKAGE)],
                       complete + [dict(Action="fail", Package=PACKAGE, Test="TestJourney/future_fault")]]
            for vector in vectors:
                write_report(report, vector)
                with self.subTest(vector=vector), self.assertRaises(ValueError):
                    gate.verify(report, "postgres")
            for malformed in ('not json\n', '[]\n', '{"Action":"run","Action":"pass"}\n', '{"Action":1}\n', 'x' * 65537):
                report.write_text(malformed)
                with self.subTest(malformed=malformed[:40]), self.assertRaises(ValueError):
                    gate.verify(report, "postgres")

    def test_matrix_rejects_empty_stale_and_unmapped_acceptance(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "matrix.yml"
            with mock.patch.object(gate, "MATRIX", path):
                path.write_text(json.dumps(selection()))
                gate.matrix()
                for field in ("postgres", "process", "harness", "coverage"):
                    bad = selection()
                    bad[field] = {}
                    path.write_text(json.dumps(bad))
                    with self.subTest(field=field), self.assertRaises(ValueError):
                        gate.matrix()
                bad = selection()
                bad["harness"]["test_audit_iterative_gate"] = ["AuditIterativeGateTest.test_absent"]
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "undeclared"):
                    gate.matrix()
                bad = selection()
                bad["coverage"]["A1"][0]["case"] = "TestAbsent"
                path.write_text(json.dumps(bad))
                with self.assertRaisesRegex(ValueError, "unselected"):
                    gate.matrix()
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
            for boundary in ("postgres", "process"):
                with mock.patch.dict(os.environ, {"CONTRACTOR_TEST_DATABASE_URL": "configured"}), mock.patch.object(gate.shutil, "which", return_value=None):
                    with self.assertRaisesRegex(ValueError, "Go"):
                        gate._prerequisites(boundary)
            with mock.patch.dict(os.environ, {"CONTRACTOR_TEST_DATABASE_URL": "configured"}), mock.patch.object(gate.shutil, "which", side_effect=lambda tool: "go" if tool == "go" else None):
                with self.assertRaisesRegex(ValueError, "Runtime"):
                    gate._prerequisites("process")

    def test_stale_evidence_is_not_reused(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate, "matrix", return_value=selection()), mock.patch.object(gate, "_prerequisites"), mock.patch.object(gate, "_source", return_value={"revision": "a" * 40, "dirty": False}):
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
            with mock.patch.object(gate, "_source", side_effect=[before, after]), mock.patch.object(gate.reports, "capture", side_effect=completed), self.assertRaisesRegex(ValueError, "source"):
                gate.run("postgres", Path(temporary))
            result = json.loads(next(Path(temporary).glob("*/result.json")).read_text())
            self.assertEqual(result["status"], "failed")

    def test_bounded_redacted_evidence_records_source_revision(self):
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            executable = directory / "go"
            payload = events()
            executable.write_text(f"#!{sys.executable}\nimport json,os\nprint(json.dumps({{'Action':'output','Package':{PACKAGE!r},'Output':os.environ['CONTRACTOR_TEST_DATABASE_URL']}}))\nfor event in {payload!r}: print(json.dumps(event))\n")
            executable.chmod(0o700)
            manifest = directory / "matrix.yml"
            manifest.write_text(json.dumps(selection()))
            env = {"PATH": str(directory) + os.pathsep + os.environ.get("PATH", ""), "CONTRACTOR_TEST_DATABASE_URL": "postgres://u:secret-value@localhost/db"}
            with mock.patch.dict(os.environ, env), mock.patch.object(gate, "MATRIX", manifest), contextlib.redirect_stdout(io.StringIO()) as output:
                result = gate.run("postgres", directory / "evidence")
            document = json.loads(result.read_text())
            self.assertEqual(document["status"], "passed")
            self.assertRegex(document["source"]["revision"], r"^[0-9a-f]{40}$")
            self.assertIsInstance(document["source"]["dirty"], bool)
            self.assertEqual(document["counts"], {"required": 1, "passed": 1, "skipped": 0})
            self.assertLessEqual(result.stat().st_size, gate.MAX_EVIDENCE_BYTES)
            self.assertNotIn("secret-value", result.read_text() + output.getvalue() + (result.parent / "go.jsonl").read_text())

    def test_process_failure_and_cancellation_record_incomplete_verification(self):
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(gate, "_prerequisites"), mock.patch.object(gate, "_source", return_value={"revision": "a" * 40, "dirty": False}), mock.patch.object(gate, "matrix", return_value=selection()), mock.patch.object(gate, "MATRIX", Path(__file__)), contextlib.redirect_stdout(io.StringIO()):
            for error in (KeyboardInterrupt(), TimeoutError(), ValueError()):
                with mock.patch.object(gate.reports, "capture", side_effect=error), self.assertRaises(type(error)):
                    gate.run("process", Path(temporary))
            statuses = {json.loads(path.read_text())["status"] for path in Path(temporary).glob("*/result.json")}
            self.assertEqual(statuses, {"cancelled", "timed_out", "failed"})


if __name__ == "__main__":
    unittest.main()
