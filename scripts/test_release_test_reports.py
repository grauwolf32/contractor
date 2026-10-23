"""Canonical report reuse must preserve the focused gates' execution evidence."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import release_test_reports as reports


class ReportReuseTest(unittest.TestCase):
    def runtime_report(self) -> ET.Element:
        suite = ET.Element("testsuite")
        audit = reports.gate("test-audit-completion-e2e.py")
        findings = reports.gate("test-findings-e2e.py")
        for identity, count in audit["matrix"]()["python"].items():
            classname, name = identity.rsplit("::", 1)
            for index in range(count):
                ET.SubElement(suite, "testcase", classname=classname, name=f"{name}[{index}]")
        classname = findings["PYTHON_FILES"][0].removesuffix(".py").replace("/", ".")
        for name, count in findings["PYTHON_CASES"].items():
            for index in range(count):
                ET.SubElement(suite, "testcase", classname=classname, name=f"{name}[{index}]")
        # Optional tests outside both focused scopes did not fail their old gates.
        ET.SubElement(
            ET.SubElement(suite, "testcase", classname="tests.optional", name="test_optional"),
            "skipped",
        )
        return suite

    def test_full_runtime_report_retains_counts_and_skip_rejection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.xml"
            suite = self.runtime_report()
            ET.ElementTree(suite).write(path)
            reports.verify_runtime(path)
            suite.remove(suite[0])
            ET.ElementTree(suite).write(path)
            with self.assertRaisesRegex(Exception, "mandatory Python"):
                reports.verify_runtime(path)
            suite = self.runtime_report()
            ET.SubElement(suite[0], "skipped")
            ET.ElementTree(suite).write(path)
            with self.assertRaisesRegex(Exception, "did not pass"):
                reports.verify_runtime(path)

    def test_each_go_gate_rejects_missing_subcases_skips_and_package_pass(self) -> None:
        for kind in ("audit-completion", "findings"):
            gate = reports.gate(f"test-{kind}-e2e.py")
            required = (
                gate["matrix"]()["go"]
                if kind == "audit-completion"
                else {gate["PACKAGE"]: gate["REQUIRED_TESTS"]}
            )
            events = []
            for package, names in required.items():
                for name in names:
                    for action in ("run", "pass"):
                        events.append(dict(Action=action, Package=package, Test=name))
                events.append(dict(Action="pass", Package=package))
            events.append(dict(Action="skip", Package="other/package", Test="TestOptional"))
            with self.subTest(kind=kind), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "go.jsonl"

                def write(items):
                    path.write_text("".join(json.dumps(event) + "\n" for event in items))

                write(events)
                reports.verify_go(path, kind)
                for changed in (
                    events[2:],
                    [dict(events[0], Action="skip"), *events[1:]],
                    [event for event in events if event.get("Test") is not None],
                ):
                    write(changed)
                    with self.assertRaises(Exception):
                        reports.verify_go(path, kind)
                path.write_text("not JSON\n")
                with self.assertRaises(ValueError):
                    reports.verify_go(path, kind)

    def test_capture_truncates_old_evidence_and_propagates_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            path.write_text("stale successful evidence\n")
            with contextlib.redirect_stdout(io.StringIO()):
                status = reports.capture(
                    path, [sys.executable, "-c", "print('new evidence'); raise SystemExit(7)"]
                )
            self.assertEqual(status, 7)
            self.assertEqual(path.read_text(), "new evidence\n")

    def test_capture_redacts_output_without_changing_framework_identity(self) -> None:
        event = dict(
            Action="pass",
            Package="contractor",
            Test="TestCase",
            Output="secret-password",
            extra="secret-password",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            from unittest import mock

            with mock.patch.dict(
                os.environ,
                {"CONTRACTOR_TEST_DATABASE_URL": "postgres://u:secret-password@localhost/db"},
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    self.assertEqual(
                        reports.capture(
                            path, [sys.executable, "-c", f"print({json.dumps(event)!r})"]
                        ),
                        0,
                    )
            self.assertNotIn("secret-password", path.read_text() + output.getvalue())
            saved = json.loads(path.read_text())
            self.assertEqual(saved["Package"], "contractor")
            self.assertEqual(saved["Action"], "pass")

    def test_cancellation_joins_the_owned_child(self) -> None:
        import signal
        import subprocess

        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "child.pid"
            child = (
                "import os, signal; from pathlib import Path; "
                f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                "os.kill(os.getppid(), signal.SIGTERM); signal.pause()"
            )
            result = subprocess.run(
                [
                    sys.executable,
                    str(Path(reports.__file__)),
                    "run",
                    "--report",
                    str(Path(directory) / "log"),
                    "--",
                    sys.executable,
                    "-c",
                    child,
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 130, result.stderr)
            self.assertIn("interrupted", result.stderr)
            pid = int(pidfile.read_text())
            try:
                with self.assertRaises(ProcessLookupError):
                    os.kill(pid, 0)
            finally:
                try:
                    os.kill(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass

    def test_non_event_json_diagnostics_are_redacted(self) -> None:
        from unittest import mock

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "diagnostics.jsonl"
            with mock.patch.dict(
                os.environ,
                {"CONTRACTOR_TEST_DATABASE_URL": "postgres://u:secret-password@localhost/db"},
            ):
                output = io.StringIO()
                with contextlib.redirect_stdout(output):
                    status = reports.capture(
                        path,
                        [
                            sys.executable,
                            "-c",
                            'print(\'{"error":"secret-password"}\'); print(\'"secret-password"\')',
                        ],
                    )
            self.assertEqual(status, 0)
            self.assertNotIn("secret-password", path.read_text() + output.getvalue())
