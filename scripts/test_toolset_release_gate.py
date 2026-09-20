"""Fail-closed verification of the release evidence parser, with no infrastructure."""

import json
import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

GATE = runpy.run_path(
    str(Path(__file__).with_name("test-toolset-runtime-configuration-release.py"))
)


class MandatoryEvidenceTests(unittest.TestCase):
    def test_missing_database_fails_before_writing_success_evidence(self):
        with (
            tempfile.TemporaryDirectory() as directory,
            patch.dict("os.environ", {}, clear=True),
        ):
            target = Path(directory) / "evidence.json"
            with self.assertRaises(GATE["GateError"]):
                GATE["run"](Path(directory) / "reports", target)
            self.assertFalse(target.exists())

    def test_go_rejects_missing_skipped_and_failed_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "go.jsonl"
            for events in (
                [],
                [{"Action": "pass", "Package": "p", "Test": "required"}],
                [{"Action": "skip", "Package": "p", "Test": "required"}],
                [{"Action": "fail", "Package": "p", "Test": "required"}],
            ):
                report.write_text("\n".join(map(json.dumps, events)))
                with self.assertRaises(GATE["GateError"]):
                    GATE["verify_go"](report, {"p": ["required"]})
            report.write_text(
                "\n".join(
                    json.dumps({"Action": action, "Package": "p", "Test": "required"})
                    for action in ("run", "pass")
                )
            )
            self.assertEqual(
                GATE["verify_go"](report, {"p": ["required"]}), ["p/required"]
            )

    def test_junit_requires_each_parametrized_case_and_no_skip(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "runtime.xml"
            report.write_text(
                '<testsuite><testcase classname="tests.test_mcp" name="test_call[tools/call]"/></testsuite>'
            )
            GATE["verify_junit"](report, ["tests/test_mcp.py::test_call[tools/call]"])
            with self.assertRaises(GATE["GateError"]):
                GATE["verify_junit"](
                    report,
                    [
                        "tests/test_mcp.py::test_call[tools/call]",
                        "tests/test_mcp.py::test_call[auth]",
                    ],
                )
            report.write_text(
                '<testsuite><testcase classname="tests.test_mcp" name="test_call[tools/call]"><skipped/></testcase></testsuite>'
            )
            with self.assertRaises(GATE["GateError"]):
                GATE["verify_junit"](
                    report, ["tests/test_mcp.py::test_call[tools/call]"]
                )

    def test_browser_rejects_summary_only_skips_retries_and_wrong_case(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "browser.json"
            passing = {
                "stats": {"expected": 1},
                "suites": [
                    {
                        "specs": [
                            {
                                "title": "required",
                                "ok": True,
                                "tests": [
                                    {
                                        "expectedStatus": "passed",
                                        "results": [{"status": "passed"}],
                                    }
                                ],
                            }
                        ]
                    }
                ],
            }
            report.write_text(json.dumps(passing))
            self.assertEqual(GATE["verify_browser"](report, ["required"]), ["required"])
            for edit in (
                lambda value: value.update(suites=[]),
                lambda value: value["stats"].update(skipped=1),
                lambda value: value["suites"][0]["specs"][0].update(title="wrong"),
                lambda value: value["suites"][0]["specs"][0]["tests"][0][
                    "results"
                ].append({"status": "passed"}),
            ):
                value = json.loads(json.dumps(passing))
                edit(value)
                report.write_text(json.dumps(value))
                with self.assertRaises(GATE["GateError"]):
                    GATE["verify_browser"](report, ["required"])
