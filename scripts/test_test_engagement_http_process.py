"""Conformance reports must contain the actual required cases, not only suite success."""

import importlib.util
import json
import tempfile
import unittest
import xml.etree.ElementTree as ET
from pathlib import Path
from unittest.mock import patch

path = Path(__file__).with_name("test-engagement-http-process.py")
spec = importlib.util.spec_from_file_location("engagement_http_gate", path)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


def go_events(required):
    events = []
    for package, names in required.items():
        parents = sorted({name.split("/", 1)[0] for name in names})
        for parent in parents:
            events.append({"Action": "run", "Package": package, "Test": parent})
            children = sorted({name for name in names if name.startswith(parent + "/")})
            events.extend({"Action": "run", "Package": package, "Test": name} for name in children)
            events.extend({"Action": "pass", "Package": package, "Test": name}
                          for name in reversed(children))
            events.append({"Action": "pass", "Package": package, "Test": parent})
        events.append({"Action": "pass", "Package": package})
    return events


class EngagementHTTPGateTests(unittest.TestCase):
    def verify_events(self, events, verify):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "go.jsonl"
            report.write_text("".join(json.dumps(event) + "\n" for event in events))
            return verify(report)

    def test_process_report_requires_every_declared_subcase(self):
        required = gate.process_cases()
        events = go_events(required)
        self.verify_events(events, gate.verify_process_report)
        for missing in required[gate.PACKAGE]:
            with self.subTest(missing=missing), self.assertRaises(gate.GateError):
                self.verify_events([event for event in events if event.get("Test") != missing],
                                   gate.verify_process_report)

    def test_process_fail_skip_and_empty_selection_are_not_success(self):
        events = go_events(gate.process_cases())
        for outcome in ("skip", "fail"):
            changed = [dict(event, Action=outcome) if event.get("Test", "").endswith("/cancel_joins")
                       and event["Action"] == "pass" else event for event in events]
            with self.subTest(outcome=outcome), self.assertRaises(gate.GateError):
                self.verify_events(changed, gate.verify_process_report)
        with self.assertRaises(gate.GateError):
            self.verify_events([{"Action": "pass", "Package": gate.PACKAGE}],
                               gate.verify_process_report)

    def test_accounting_requires_both_real_store_owners(self):
        required = gate.matrix()["accounting"]
        events = go_events(required)
        self.verify_events(events, gate.verify_accounting_report)
        for package in required:
            with self.subTest(package=package), self.assertRaises(gate.GateError):
                self.verify_events([event for event in events if event["Package"] != package],
                                   gate.verify_accounting_report)

    def python_report(self):
        suite = ET.Element("testsuite")
        for identity in gate.matrix()["python"]:
            module, name = identity.rsplit("::", 1)
            ET.SubElement(suite, "testcase", classname=module, name=name)
        ET.SubElement(ET.SubElement(suite, "testcase", classname="tests.optional",
                                   name="test_optional"), "skipped")
        return suite

    def verify_python(self, suite):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "runtime.xml"
            ET.ElementTree(suite).write(report)
            return gate.verify_python_report(report)

    def test_python_requires_exact_parameter_identity_not_only_case_count(self):
        self.verify_python(self.python_report())
        suite = self.python_report()
        suite[0].set("name", "test_renamed_or_wrong_parameter")
        with self.assertRaisesRegex(gate.GateError, "mandatory Python"):
            self.verify_python(suite)
        suite = self.python_report()
        suite.append(suite[0])
        with self.assertRaisesRegex(gate.GateError, "duplicate"):
            self.verify_python(suite)

    def test_python_rejects_any_failure_or_skip_within_owned_module(self):
        for tag in ("skipped", "failure", "error"):
            suite = self.python_report()
            extra = ET.SubElement(suite, "testcase", classname=suite[0].get("classname"),
                                  name="test_additional_conformance_case")
            ET.SubElement(extra, tag)
            with self.subTest(tag=tag), self.assertRaises(gate.GateError):
                self.verify_python(suite)

    def test_matrix_rejects_missing_empty_and_duplicate_selections(self):
        valid = gate.matrix()
        with tempfile.TemporaryDirectory() as directory:
            matrix_path = Path(directory) / "matrix.json"
            for key in valid:
                changed = dict(valid, **{key: [] if key == "python" else {}})
                matrix_path.write_text(json.dumps(changed))
                with patch.object(gate, "MATRIX", matrix_path), self.assertRaises(gate.GateError):
                    gate.matrix()
            changed = dict(valid, python=[valid["python"][0], valid["python"][0]])
            matrix_path.write_text(json.dumps(changed))
            with patch.object(gate, "MATRIX", matrix_path), self.assertRaises(gate.GateError):
                gate.matrix()


if __name__ == "__main__":
    unittest.main()
