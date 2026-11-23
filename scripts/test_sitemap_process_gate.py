"""The sitemap entry point requires complete process and child-case reports."""

import contextlib
import io
import json
import os
import runpy
import unittest
from pathlib import Path
from unittest.mock import patch

import release_test_reports as reports

GATE = runpy.run_path(str(Path(__file__).with_name("test-project-sitemap-tools-e2e.py")))
PACKAGE = "github.com/grauwolf32/contractor/tests/e2e"


def complete_events():
    events = []
    for root in GATE["PROCESS_TESTS"]:
        events.append({"Action": "run", "Package": PACKAGE, "Test": root})
        for case in GATE["REQUIRED_TESTS"]:
            if case.startswith(root + "/"):
                events.extend({"Action": action, "Package": PACKAGE, "Test": case}
                              for action in ("run", "pass"))
        events.append({"Action": "pass", "Package": PACKAGE, "Test": root})
    events.append({"Action": "pass", "Package": PACKAGE})
    return events


class SitemapProcessGateTests(unittest.TestCase):
    def run_gate(self, events, returncode=0):
        def capture(report, command, **options):
            report.write_text("".join(json.dumps(event) + "\n" for event in events))
            self.assertIn(reports.pattern(GATE["PROCESS_TESTS"]), command)
            self.assertGreater(options["max_report_bytes"], 0)
            return returncode

        with patch.dict(os.environ, {"CONTRACTOR_TEST_DATABASE_URL": "synthetic"}), \
                patch.object(reports, "capture", side_effect=capture), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            GATE["main"]()
        return output.getvalue()

    def test_complete_report_passes(self):
        self.assertIn("8 passing cases, zero skips", self.run_gate(complete_events()))

    def test_required_children_cannot_be_missing_or_skipped(self):
        for child in GATE["REQUIRED_TESTS"][len(GATE["PROCESS_TESTS"]):]:
            for outcome in ("missing", "skip", "fail"):
                events = [event for event in complete_events()
                          if outcome != "missing" or event.get("Test") != child]
                if outcome != "missing":
                    events = [dict(event, Action=outcome)
                              if event.get("Test") == child and event["Action"] == "pass"
                              else event for event in events]
                with self.subTest(child=child, outcome=outcome), \
                        self.assertRaises(reports.GateError):
                    self.run_gate(events)

    def test_package_completion_and_process_success_are_required(self):
        with self.assertRaises(reports.GateError):
            self.run_gate(complete_events()[:-1])
        with self.assertRaises(SystemExit) as error:
            self.run_gate(complete_events(), returncode=1)
        self.assertEqual(error.exception.code, 1)
