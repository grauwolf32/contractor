"""Regression checks for accepting incomplete Go JSON reports."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location("composition_gate", Path(__file__).with_name("test-audit-composition-e2e.py"))
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class CompositionGateTest(unittest.TestCase):
    def test_incomplete_or_skipped_selection_cannot_pass(self):
        required = {"example/package": ["TestJourney/required"]}
        complete = [
            {"Package": "example/package", "Test": case, "Action": action}
            for action, case in (("run", "TestJourney"), ("run", "TestJourney/required"),
                                 ("pass", "TestJourney/required"), ("pass", "TestJourney"))
        ] + [{"Package": "example/package", "Action": "pass"}]
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "go.jsonl"

            def write(events):
                report.write_text("".join(json.dumps(event) + "\n" for event in events))

            write(complete)
            self.assertEqual(gate.verify(report, required), 1)
            for events in ([], complete[:2] + complete[-1:], complete[:-1],
                           complete + [{"Package": "example/package", "Test": "TestJourney/required", "Action": "skip"}],
                           complete + [{"Package": "example/package", "Action": "fail"}]):
                with self.subTest(events=events):
                    write(events)
                    with self.assertRaises(ValueError):
                        gate.verify(report, required)

class CompositionEvidenceTest(unittest.TestCase):
    def test_retry_invalidates_old_pass_on_prerequisite_matrix_and_journey_failure(self):
        import contextlib
        import io
        import os
        import sys
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            output = root / 'evidence'
            output.mkdir()
            matrix = root / 'matrix.json'
            required = {'example/package': ['TestJourney/required']}
            document = {'integration': required, 'process': required}
            for dependency in ('runtime/.venv/bin/python', 'ui/node_modules/@playwright/test/cli.js'):
                path = root / dependency
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            environment = {'CONTRACTOR_TEST_DATABASE_URL': 'configured',
                           'CONTRACTOR_AUDIT_COMPOSITION_EVIDENCE_DIR': str(output)}
            for failure in ('database', 'tool', 'matrix', 'journey', 'cancelled', 'timed_out'):
                matrix.write_text(json.dumps(document) if failure != 'matrix' else 'not json')
                (output / 'gate.json').write_text('{"status":"passed"}')
                (output / 'journey.json').write_text('{"passed":true}')
                def execute(*_):
                    if failure == 'cancelled':
                        raise KeyboardInterrupt
                    if failure == 'timed_out':
                        raise TimeoutError
                    return ['example/package/TestJourney', 'example/package/TestJourney/required']
                configured = dict(environment)
                if failure == 'database':
                    configured['CONTRACTOR_TEST_DATABASE_URL'] = ''
                with (self.subTest(failure=failure), mock.patch.dict(os.environ, configured),
                      mock.patch.object(gate, 'ROOT', root), mock.patch.object(gate, 'MATRIX', matrix),
                      mock.patch.object(gate.shutil, 'which', return_value=None if failure == 'tool' else 'tool'),
                      mock.patch.object(gate.reports, 'source_state', return_value={'revision': 'a' * 40, 'dirty': False}),
                      mock.patch.object(gate, 'execute', side_effect=execute) as command,
                      mock.patch.object(sys, 'argv', ['composition']), contextlib.redirect_stdout(io.StringIO())):
                    with self.assertRaises(BaseException):
                        gate.main()
                    if failure in ('database', 'tool', 'matrix'):
                        command.assert_not_called()
                    result = json.loads((output / 'gate.json').read_text())
                    expected = failure if failure in ('cancelled', 'timed_out') else 'failed'
                    self.assertEqual(result['status'], expected)
                    self.assertNotIn('mandatoryCases', result)

    def test_report_only_never_replaces_summary_or_pins_new_source(self):
        import sys
        from unittest import mock

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            report = root / 'go.jsonl'
            required = {'example/package': ['TestJourney']}
            report.write_text(''.join(json.dumps(dict(Action=action, Package='example/package', Test=case)) + '\n'
                                      for action, case in (('run', 'TestJourney'), ('pass', 'TestJourney'), ('pass', ''))))
            with (mock.patch.object(gate, 'matrix', return_value={'process': required}),
                  mock.patch.object(gate.reports, 'source_state') as source,
                  mock.patch.object(gate.reports, 'write_evidence') as write,
                  mock.patch.object(gate, 'execute') as execute,
                  mock.patch.object(sys, 'argv', ['composition', '--verify-only', str(report)])):
                gate.main()
                source.assert_not_called()
                write.assert_not_called()
                execute.assert_not_called()


if __name__ == "__main__":
    unittest.main()
