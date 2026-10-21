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
            for case in ("TestJourney", "TestJourney/required") for action in ("run", "pass")
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


if __name__ == "__main__":
    unittest.main()
