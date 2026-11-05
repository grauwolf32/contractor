"""Disk-heavy fixtures stay opt-in, including full-gate report verification."""

import contextlib
import io
import json
import os
import re
import runpy
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import disk_heavy_policy as policy
import release_test_reports as reports
import shared_fixture_gate as shared

ROOT = Path(__file__).resolve().parents[1]
PACKAGE, HEAVY = next(iter(policy.CASES))
PACKAGE = policy.MODULE + PACKAGE


def events(heavy="skip"):
    result = [
        {"Action": "run", "Package": PACKAGE, "Test": HEAVY},
        {"Action": heavy, "Package": PACKAGE, "Test": HEAVY},
        {"Action": "run", "Package": PACKAGE, "Test": "TestOrdinary"},
        {"Action": "pass", "Package": PACKAGE, "Test": "TestOrdinary"},
        {"Action": "pass", "Package": PACKAGE},
    ]
    return result


class DiskPolicyTest(unittest.TestCase):
    def test_flag_requires_explicit_one(self):
        for value, expected in (("", False), ("0", False), ("1", True)):
            with (
                self.subTest(value=value),
                mock.patch.dict(os.environ, {policy.FLAG: value}),
            ):
                self.assertEqual(policy.enabled(), expected)
        with (
            mock.patch.dict(os.environ, {policy.FLAG: "true"}),
            self.assertRaises(ValueError),
        ):
            policy.enabled()

    def test_registry_equals_guards_before_any_fixture_setup(self):
        found = set()
        for path in (ROOT / "internal").rglob("*_test.go"):
            text = path.read_text()
            guards = re.findall(
                r"^func (Test\w+)\(t \*testing.T\) \{\n\tdisktest.Require\(t\)",
                text,
                re.MULTILINE,
            )
            self.assertEqual(text.count("disktest.Require(t)"), len(guards), str(path))
            found.update(
                (path.parent.relative_to(ROOT).as_posix(), name) for name in guards
            )
        self.assertEqual(found, set(policy.CASES))
        # Neither full Make goals nor CI enables the flag implicitly.
        make_sources = (ROOT / "Makefile").read_text() + "".join(
            path.read_text() for path in (ROOT / "make").glob("*.mk")
        )
        self.assertNotIn(policy.FLAG, make_sources)
        workflow = (ROOT / ".github/workflows/ci.yml").read_text()
        self.assertIn(
            "default: false", workflow.split("heavy_io:", 1)[1].split("jobs:", 1)[0]
        )
        self.assertIn(
            "github.event_name == 'workflow_dispatch' && inputs.heavy_io", workflow
        )

    def verify(self, stream):
        with (
            tempfile.TemporaryDirectory() as directory,
            contextlib.redirect_stdout(io.StringIO()),
        ):
            report = Path(directory) / "go.jsonl"
            report.write_text("".join(json.dumps(event) + "\n" for event in stream))
            return reports.verify_go_report(report, {PACKAGE: [HEAVY, "TestOrdinary"]})

    def test_disabled_skip_is_not_passed_coverage(self):
        with mock.patch.dict(os.environ, {policy.FLAG: "0"}):
            self.assertEqual(self.verify(events()), [PACKAGE + "/TestOrdinary"])
            # Optionality never hides failure, missing tests or an undeclared skip.
            for stream in (
                events("fail"),
                events()[2:],
                [*events()[:2], *events()[3:]],
                [
                    *events()[:3],
                    {"Action": "skip", "Package": PACKAGE, "Test": "TestOrdinary"},
                    events()[-1],
                ],
                [events()[1], *events()[2:]],
                [*events()[:2], events()[1], *events()[2:]],
            ):
                with self.assertRaises(reports.GateError):
                    self.verify(stream)

    def test_opt_in_requires_pass_even_in_full_report(self):
        with mock.patch.dict(os.environ, {policy.FLAG: "1"}):
            with self.assertRaises(reports.GateError):
                self.verify(events())
            self.assertIn(PACKAGE + "/" + HEAVY, self.verify(events("pass")))

    def test_shared_fixture_gate_accepts_only_explicit_disabled_skip(self):
        key = (PACKAGE.removeprefix(policy.MODULE), HEAVY)
        discovered = {
            key: "source.go",
            ("internal/light", "TestSharedLight"): "light.go",
        }
        outcomes = {key: "skip", ("internal/light", "TestSharedLight"): "pass"}
        with mock.patch.dict(os.environ, {policy.FLAG: "0"}):
            self.assertEqual(shared.check_go_outcomes(discovered, outcomes), [])
            self.assertTrue(
                shared.check_go_outcomes(
                    discovered, {("internal/light", "TestSharedLight"): "pass"}
                )
            )
        with mock.patch.dict(os.environ, {policy.FLAG: "1"}):
            self.assertTrue(shared.check_go_outcomes(discovered, outcomes))

    def test_iterative_evidence_records_skip_instead_of_pass(self):
        gate = runpy.run_path(str(ROOT / "scripts/test-audit-iterative.py"))

        def captured(path, _command, **_kwargs):
            path.write_text("".join(json.dumps(event) + "\n" for event in events()))
            return 0

        with (
            tempfile.TemporaryDirectory() as directory,
            mock.patch.dict(os.environ, {policy.FLAG: "0"}),
            mock.patch.dict(
                gate["run"].__globals__,
                {
                    "matrix": lambda: {"postgres": {PACKAGE: [HEAVY, "TestOrdinary"]}},
                    "_prerequisites": lambda _: None,
                    "MATRIX": Path(__file__),
                },
            ),
            mock.patch.object(
                reports,
                "source_state",
                return_value={"revision": "a" * 40, "dirty": False},
            ),
            mock.patch.object(reports, "capture", side_effect=captured),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            document = json.loads(gate["run"]("postgres", Path(directory)).read_text())
        self.assertEqual(document["counts"], {"required": 1, "passed": 1, "skipped": 1})
        self.assertEqual(document["cases"], [PACKAGE + "/TestOrdinary"])
        self.assertEqual(document["optional_disk_heavy_skips"], [PACKAGE + "/" + HEAVY])

    def test_sitemap_gate_keeps_missing_and_failure_checks(self):
        gate = runpy.run_path(str(ROOT / "scripts/test-project-sitemap-postgres.py"))
        volume = "TestPostgresSitemapSingleRunExceedsRemovedPublicationCaps"

        def fake_run(command, *, stdout, **_kwargs):
            for package, case in sorted(gate["REQUIRED_CASES"]):
                for action in ("run", "skip" if case == volume else "pass"):
                    stdout.write(
                        json.dumps({"Package": package, "Test": case, "Action": action})
                        + "\n"
                    )
            for package in gate["PACKAGES"]:
                stdout.write(
                    json.dumps({"Package": policy.MODULE + package, "Action": "pass"})
                    + "\n"
                )
            return subprocess.CompletedProcess(command, 0)

        with (
            mock.patch.dict(
                os.environ,
                {policy.FLAG: "0", "CONTRACTOR_TEST_DATABASE_URL": "configured"},
            ),
            mock.patch.object(subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            gate["main"]()
        with (
            mock.patch.dict(
                os.environ,
                {policy.FLAG: "1", "CONTRACTOR_TEST_DATABASE_URL": "configured"},
            ),
            mock.patch.object(subprocess, "run", side_effect=fake_run),
            contextlib.redirect_stdout(io.StringIO()),
            self.assertRaises(SystemExit),
        ):
            gate["main"]()
