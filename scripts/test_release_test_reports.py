"""Canonical report reuse must preserve the focused gates' execution evidence."""

import contextlib
import io
import json
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path

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
        pentest = reports.gate("test-pentest-http-process.py")
        for identity in pentest["matrix"]()["python"]:
            classname, name = identity.rsplit("::", 1)
            ET.SubElement(suite, "testcase", classname=classname, name=name)
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
                all_names = {"/".join(name.split("/")[:length]) for name in names for length in range(1, len(name.split("/")) + 1)}
                for action, names in (("run", sorted(all_names)), ("pass", sorted(all_names, reverse=True))):
                    events.extend(dict(Action=action, Package=package, Test=name) for name in names)
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

    def test_shared_go_verifier_preserves_each_family_root_scope(self) -> None:
        package = "example/audit"
        required = {package: ["TestAuditCompletion/required"]}
        valid = [dict(Action=action, Package=package, Test=case) for action, case in (
            ("run", "TestAuditCompletion"), ("run", "TestAuditCompletion/required"),
            ("pass", "TestAuditCompletion/required"), ("pass", "TestAuditCompletion"))]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "go.jsonl"
            def report(extra):
                path.write_text("".join(json.dumps(event) + "\n" for event in [
                    *valid, *extra, dict(Action="pass", Package=package)]))
            for action in ("skip", "fail", "build-fail"):
                report([dict(Action=action, Package=package, Test="TestOptional")])
                reports.verify_go_report(path, required)
                with self.assertRaises(ValueError):
                    reports.verify_go_report(path, required, root_pattern=".*")
                reports.verify_go_report(path, required, root_pattern="AuditCompletion|^TestImporter|^TestStandard")
                for root in ("TestImporterUnlisted", "TestStandardUnlisted", "TestAuditCompletion/unlisted"):
                    report([dict(Action=action, Package=package, Test=root)])
                    with self.assertRaises(ValueError):
                        reports.verify_go_report(path, required, root_pattern="AuditCompletion|^TestImporter|^TestStandard")
                report([dict(Action=action, Package=package)])
                with self.assertRaises(ValueError):
                    reports.verify_go_report(path, required)
            report([dict(Action="run", Package=package, Test="TestImporterExtra"),
                    dict(Action="pass", Package=package, Test="TestImporterExtra")])
            self.assertIn(package + "/TestImporterExtra", reports.verify_go_report(
                path, required, root_pattern="AuditCompletion|^TestImporter|^TestStandard"))

    def test_literal_slashes_do_not_invent_intermediate_parent_tests(self) -> None:
        package = "example/audit"
        names = ("TestRoot/x", "TestRoot/x/y", "TestRoot/virtual/middle/leaf")
        events = [dict(Action="run", Package=package, Test="TestRoot")]
        for name in names:
            events.extend(dict(Action=action, Package=package, Test=name) for action in ("run", "pass"))
        events += [dict(Action="pass", Package=package, Test="TestRoot"), dict(Action="pass", Package=package)]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "go.jsonl"
            report.write_text("".join(json.dumps(event) + "\n" for event in events))
            self.assertEqual(reports.verify_go_report(report, {package: names}),
                             sorted(package + "/" + name for name in ("TestRoot", *names)))
            # The guaranteed root still cannot pass before a required or running subcase.
            early = [events[0], events[-2], *events[1:-2], events[-1]]
            report.write_text("".join(json.dumps(event) + "\n" for event in early))
            with self.assertRaises(ValueError):
                reports.verify_go_report(report, {package: names})

    def test_diagnostic_console_keeps_credentials_out_of_framework_identity(self) -> None:
        from unittest import mock
        for password in ("pass", "contractor"):
            event = dict(Action="pass", Package="contractor", Test="TestCase", Output=password)
            with self.subTest(password=password), tempfile.TemporaryDirectory() as directory, mock.patch.dict(
                os.environ, {"CONTRACTOR_TEST_DATABASE_URL": "postgres://u:" + password + "@localhost/db"},
            ), contextlib.redirect_stdout(io.StringIO()) as displayed:
                report = Path(directory) / "go.jsonl"
                self.assertEqual(reports.capture(report, [sys.executable, "-c", "print(" + repr(json.dumps(event)) + ")"], display_events=False), 0)
                self.assertNotIn(password, displayed.getvalue())
                saved = json.loads(report.read_text())
                self.assertEqual(saved["Action"], "pass")
                self.assertEqual(saved["Package"], "contractor")
                self.assertEqual(saved["Output"], "[redacted]")

    def test_capture_truncates_old_evidence_and_propagates_failure(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "report.jsonl"
            path.write_text("stale successful evidence\n")
            with contextlib.redirect_stdout(io.StringIO()):
                status = reports.capture(
                    path, [sys.executable, "-c", "print('new evidence'); raise SystemExit(7)"]
                )
            self.assertEqual(status, 7)
            self.assertEqual(path.read_text(), "Invalid test event: new evidence\n")

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
        from unittest import mock

        from test_audit_iterative_gate import gate as iterative
        from test_audit_iterative_gate import selection
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

        from test_audit_iterative_gate import events, selection, write_report
        from test_audit_iterative_gate import gate as iterative
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(iterative, "matrix", return_value=selection()):
            path = Path(directory) / "go.jsonl"
            write_report(path, events())
            module = {"matrix": iterative.matrix, "verify": iterative.verify}
            with mock.patch.object(reports, "gate", return_value=module), mock.patch.object(reports, "capture") as capture, mock.patch.object(iterative, "run") as execute, mock.patch.object(reports, "source_state") as source:
                for action in ("verify-iterative-postgres", "verify-iterative-process"):
                    with self.subTest(action=action), mock.patch.object(sys, "argv", ["reports", action, "--report", str(path)]):
                        self.assertEqual(reports.main(), 0)
                with mock.patch.object(sys, "argv", ["reports", "iterative-pattern"]), contextlib.redirect_stdout(io.StringIO()) as output:
                    self.assertEqual(reports.main(), 0)
                capture.assert_not_called()
                execute.assert_not_called()
                source.assert_not_called()
                self.assertEqual(output.getvalue(), "TestJourney\n")
                write_report(path, [])
                with mock.patch.object(sys, "argv", ["reports", "verify-iterative-postgres", "--report", str(path)]), self.assertRaises(ValueError):
                    reports.main()

    def test_redaction_cannot_repair_a_malformed_terminal_event(self) -> None:
        from unittest import mock
        package = "example/audit"
        # The decoded quote in the password makes this terminal event invalid JSON.
        # Redaction must not turn it into valid passing execution evidence.
        lines = [json.dumps(dict(Action="run", Package=package, Test="TestRequired")),
                 '{"Action":"pass","Package":"example/audit","Test":"TestRequired","Output":"q"uote"}',
                 json.dumps(dict(Action="pass", Package=package))]
        with tempfile.TemporaryDirectory() as directory, mock.patch.dict(os.environ, {
            "CONTRACTOR_TEST_DATABASE_URL": "postgres://u:q%22uote@localhost/db",
        }), contextlib.redirect_stdout(io.StringIO()) as displayed:
            report = Path(directory) / "go.jsonl"
            self.assertEqual(reports.capture(report, [sys.executable, "-c", "print(" + repr("\n".join(lines)) + ")"]), 0)
            self.assertNotIn('q"uote', report.read_text() + displayed.getvalue())
            with self.assertRaises(ValueError):
                reports.verify_go_report(report, {package: ["TestRequired"]})

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

class AuditEvidenceTest(unittest.TestCase):
    def test_completion_pins_execution_and_never_promotes_failed_attempts(self):
        from unittest import mock

        from test_audit_iterative_gate import PACKAGE, events, write_report

        completion = reports.gate('test-audit-completion-e2e.py')
        namespace = completion['run'].__globals__
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runtime = root / 'runtime/.venv/bin/python'
            runtime.parent.mkdir(parents=True)
            runtime.touch()
            matrix = root / 'matrix.json'
            document = {'go': {PACKAGE: ['TestJourney/required']},
                        'python': {'tests.fixture::test_case': 1}}
            for disposition in ('passed', 'cancelled', 'timed_out', 'changed_matrix', 'changed_source'):
                matrix.write_text(json.dumps(document))
                state = {'revision': 'a' * 40, 'dirty': False}
                def run_command(_command, *, cwd, report):
                    if disposition == 'cancelled':
                        raise KeyboardInterrupt
                    if disposition == 'timed_out':
                        raise TimeoutError
                    if report.name == 'runtime.log':
                        report.write_text('bounded runtime log')
                        (report.parent / 'runtime.xml').write_text('<testsuite><testcase classname="tests.fixture" name="test_case"/></testsuite>')
                    else:
                        write_report(report, events())
                        if disposition == 'changed_matrix':
                            matrix.write_text(matrix.read_text() + ' ')
                        elif disposition == 'changed_source':
                            state['dirty'] = True
                output = root / disposition
                with (self.subTest(disposition=disposition),
                      mock.patch.dict(os.environ, {'CONTRACTOR_TEST_DATABASE_URL': 'configured'}),
                      mock.patch.dict(namespace, {'ROOT': root, 'MATRIX': matrix, 'run_command': run_command}),
                      mock.patch.object(namespace['shutil'], 'which', return_value='tool'),
                      mock.patch.object(reports, 'source_state', side_effect=lambda: dict(state)),
                      contextlib.redirect_stdout(io.StringIO())):
                    if disposition == 'passed':
                        completion['run'](output)
                    else:
                        with self.assertRaises(BaseException):
                            completion['run'](output)
                    result = json.loads(next(output.glob('*/executed.json')).read_text())
                    self.assertEqual(result['status'], disposition if disposition in ('passed', 'cancelled', 'timed_out') else 'failed')
                    if disposition == 'passed':
                        self.assertEqual(result['source'], {'revision': 'a' * 40, 'dirty': False})
                        self.assertEqual(result['matrix']['digest'], reports.digest(matrix))
                        self.assertEqual(result['go'], [PACKAGE + '/TestJourney', PACKAGE + '/TestJourney/required'])
                        self.assertEqual(result['python'], ['tests.fixture::test_case'])
                        self.assertEqual(set(result['reports']), {'go.jsonl', 'runtime.xml'})
                    else:
                        self.assertNotIn('go', result)
                        self.assertNotIn('python', result)

    def test_evidence_writer_is_bounded_and_atomic(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'result.json'
            reports.write_evidence(path, {'status': 'incomplete'})
            with self.assertRaises(ValueError):
                reports.write_evidence(path, {'payload': 'x' * (64 << 10)})
            self.assertEqual(json.loads(path.read_text()), {'status': 'incomplete'})
            reports.write_evidence(path, {'payload': 'x' * (80 << 10)}, max_bytes=128 << 10)
            self.assertEqual(len(json.loads(path.read_text())['payload']), 80 << 10)
