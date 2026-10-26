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

    def test_cancellation_joins_descendant_processes(self) -> None:
        import signal
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            childfile, grandfile = Path(directory) / "child", Path(directory) / "grandchild"
            grandchild = "import os,signal; from pathlib import Path; " + f"Path({str(grandfile)!r}).write_text(str(os.getpid())); signal.pause()"
            child = ("import os,signal,subprocess,sys,time; from pathlib import Path\n"
                     + f"Path({str(childfile)!r}).write_text(str(os.getpid()))\n"
                     + f"process=subprocess.Popen([sys.executable,'-c',{grandchild!r}])\n"
                     + "def stop(*_):\n process.wait(timeout=5)\n sys.exit(0)\n"
                     + "signal.signal(signal.SIGTERM,stop)\n"
                     + f"while not Path({str(grandfile)!r}).exists(): time.sleep(0.01)\n"
                     + "os.kill(os.getppid(),signal.SIGTERM)\nsignal.pause()\n")
            result = subprocess.run([sys.executable, str(Path(reports.__file__)), "run", "--report", str(Path(directory) / "report"), "--", sys.executable, "-c", child], capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 130, result.stderr)
            for path in (childfile, grandfile):
                pid = int(path.read_text())
                try:
                    with self.assertRaises(ProcessLookupError):
                        os.kill(pid, 0)
                finally:
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass

    def test_deadline_joins_silent_child(self) -> None:
        import signal
        import subprocess
        import threading
        from types import SimpleNamespace
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory) / "pid"
            child = ("import os,signal; from pathlib import Path; "
                     "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                     + f"Path({str(pidfile)!r}).write_text(str(os.getpid())); "
                     + "print('ready',flush=True); signal.pause()")
            ready, finished = threading.Event(), threading.Event()
            waits = []

            class ControlledFinished:
                def wait(self, timeout=None):
                    waits.append(timeout)
                    if len(waits) == 1:
                        # The deadline cannot expire before the real child's
                        # SIGTERM handler and PID publication are ready.
                        ready.wait()
                    return finished.is_set()

                def set(self):
                    finished.set()

            class ReadyOutput:
                def __init__(self, stream):
                    self.stream = stream

                def readline(self, *args):
                    line = self.stream.readline(*args)
                    if line == "ready\n" or not line:
                        ready.set()
                    return line

                def close(self):
                    self.stream.close()

            real_popen = subprocess.Popen
            def start(*args, **kwargs):
                process = real_popen(*args, **kwargs)
                process.stdout = ReadyOutput(process.stdout)
                return process

            supplied = iter((ControlledFinished(), threading.Event()))
            # Patch this module's bindings, not threading.Event/Popen globally:
            # Thread itself must retain its real internal synchronization.
            owned_threads = SimpleNamespace(Event=lambda: next(supplied), Thread=threading.Thread)
            owned_processes = SimpleNamespace(Popen=start, TimeoutExpired=subprocess.TimeoutExpired, PIPE=subprocess.PIPE, STDOUT=subprocess.STDOUT)
            with (mock.patch.object(reports, "threading", owned_threads),
                  mock.patch.object(reports, "subprocess", owned_processes),
                  contextlib.redirect_stdout(io.StringIO())):
                with self.assertRaises(TimeoutError):
                    reports.capture(Path(directory) / "report", [sys.executable, "-c", child], timeout_seconds=60)
            self.assertTrue(pidfile.is_file())
            self.assertEqual(waits, [60, reports.CLEANUP_GRACE_SECONDS])
            with self.assertRaises(ProcessLookupError):
                os.kill(int(pidfile.read_text()), 0)

    def test_content_free_capture_never_turns_malformed_output_into_success(self) -> None:
        from test_audit_iterative_gate import gate as iterative, selection
        from unittest import mock
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()) as output:
            report = Path(directory) / "report"
            self.assertEqual(reports.capture(report, [sys.executable, "-c", "print('postgres://u:secret-value@localhost/db fixture-target-body-marker')"], retain_output=False), 0)
            self.assertNotIn("fixture-target-body-marker", report.read_text() + output.getvalue())
            self.assertNotIn("secret-value", report.read_text() + output.getvalue())
            with mock.patch.object(iterative, "matrix", return_value=selection()), self.assertRaises(ValueError):
                iterative.verify(report, "postgres")

    def test_capture_bounds_malformed_output_and_joins(self) -> None:
        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            for text, options in (("x" * 65537, {}), ("bounded diagnostic", {"max_report_bytes": 4})):
                with self.subTest(options=options), self.assertRaises(ValueError):
                    reports.capture(Path(directory) / "report", [sys.executable, "-c", f"print({text!r})"], **options)
            for timeout in (0, -1, float("nan"), float("inf")):
                with self.subTest(timeout=timeout), self.assertRaises(ValueError):
                    reports.capture(Path(directory) / "report", [sys.executable, "-c", "pass"], timeout_seconds=timeout)

    def test_iterative_report_cli_reuses_canonical_cases(self) -> None:
        from unittest import mock
        from test_audit_iterative_gate import gate as iterative, selection, events, write_report
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(iterative, "matrix", return_value=selection()):
            path = Path(directory) / "go.jsonl"
            write_report(path, events())
            module = {"matrix": iterative.matrix, "verify": iterative.verify}
            with mock.patch.object(reports, "gate", return_value=module):
                for action in ("verify-iterative-postgres", "verify-iterative-process"):
                    with self.subTest(action=action), mock.patch.object(sys, "argv", ["reports", action, "--report", str(path)]):
                        self.assertEqual(reports.main(), 0)
                with mock.patch.object(sys, "argv", ["reports", "iterative-pattern"]), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(reports.main(), 0)
                self.assertEqual(output.getvalue(), "TestJourney\n")
                write_report(path, [])
                with mock.patch.object(sys, "argv", ["reports", "verify-iterative-postgres", "--report", str(path)]), self.assertRaises(ValueError):
                    reports.main()

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
