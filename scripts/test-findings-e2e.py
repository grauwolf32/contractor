#!/usr/bin/env python3
"""Run the offline findings gate and reject absent, skipped or incomplete test sets."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

PACKAGE = "github.com/grauwolf32/contractor/tests/e2e"
PROCESS_TESTS = (
    "TestFindingsProducerAndReaderAcrossProcesses",
    "TestFindingsCollectionsRetainOrdinaryAndAuditReceipts",
    "TestFindingsReaderBoundariesAcrossProcesses",
)
REQUIRED_TESTS = frozenset(
    (
        *PROCESS_TESTS,
        *(
            PROCESS_TESTS[1] + "/" + name
            for name in (
                "shared-function-preserves-operations",
                "ordinary-hypothesis-and-intake-replay",
                "snapshot-survives-source-deletion",
                "changed-selection-conflicts",
                "pagination-and-exact-evidence",
            )
        ),
        *(
            PROCESS_TESTS[2] + "/" + name
            for name in (
                "inaccessible-input",
                "empty-is-explicit-success",
                "interrupted-preparation-reuses-exact-ref",
                "conflicting-document-fails",
                "invalid-zip-fails",
            )
        ),
    )
)
# Minimum counts also protect parametrized writer/reader modes and malformed inputs.
PYTHON_CASES = {
    "test_finding_uses_runtime_identity_and_exact_evidence": 2,
    "test_finding_rejects_non_exact_and_duplicate_evidence": 1,
    "test_finding_transport_loss_retries_byte_identical_submission": 1,
    "test_shared_go_python_fixture_and_exact_consumer_reads": 1,
    "test_selected_tools_and_adk_descriptions": 4,
    "test_reader_filters_cursors_previews_and_defensive_copies": 1,
    "test_empty_collection_is_success": 1,
    "test_replay_loss_interruption_conflict_and_access_errors": 1,
    "test_invalid_collection_fails_before_any_materialization": 20,
    "test_archive_boundaries": 10,
    "test_byte_bounded_pagination_never_drops_items": 1,
    "test_page_item_bound_and_cursor_can_change_limit": 1,
    "test_collection_supports_audit_holds_and_noncanonical_proposals": 1,
    "test_artifact_client_enforces_reader_byte_limit_even_with_unbounded_transport": 1,
    "test_reader_preparation_distinguishes_byte_limit_from_missing_input": 1,
    "test_reader_errors_keep_distinct_codes_in_model_visible_envelope": 1,
}


class GateError(Exception):
    pass


def verify_go_report(path: Path) -> None:
    started, passed = set(), set()
    package_pass = False
    for line in path.read_text().splitlines():
        try:
            event = json.loads(line)
        except (ValueError, TypeError) as error:
            raise GateError("Go test output is not a JSON event stream") from error
        if not isinstance(event, dict) or event.get("Package") != PACKAGE:
            continue
        action, test = event.get("Action"), event.get("Test")
        if action in {"skip", "fail"}:
            raise GateError(f"Go findings test did not pass: {test or PACKAGE} ({action})")
        if test:
            if action == "run":
                started.add(test)
            elif action == "pass":
                passed.add(test)
        elif action == "pass":
            package_pass = True
    missing = REQUIRED_TESTS - (started & passed)
    if missing or not package_pass:
        raise GateError(
            "mandatory Go findings cases did not execute and pass: "
            + ", ".join(sorted(missing or {PACKAGE}))
        )


def verify_python_report(path: Path) -> None:
    try:
        root = ET.parse(path).getroot()
    except (ET.ParseError, OSError) as error:
        raise GateError("pytest did not produce a valid JUnit report") from error
    counts: dict[str, int] = {}
    for case in root.iter("testcase"):
        name = case.get("name", "").split("[", 1)[0]
        if any(case.find(tag) is not None for tag in ("skipped", "failure", "error")):
            raise GateError("Python findings test did not pass: " + name)
        counts[name] = counts.get(name, 0) + 1
    missing = [name for name, minimum in PYTHON_CASES.items() if counts.get(name, 0) < minimum]
    if missing:
        raise GateError(
            "mandatory Python findings cases did not execute and pass: "
            + ", ".join(sorted(missing))
        )


def run(root: Path) -> None:
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL", "").strip():
        raise GateError("CONTRACTOR_TEST_DATABASE_URL is required; use disposable PostgreSQL")
    python = root / "runtime/.venv/bin/python"
    if not python.is_file() or shutil.which("go") is None:
        raise GateError("Go and the prepared runtime/.venv/bin/python are required")
    with tempfile.TemporaryDirectory(prefix="contractor-findings-gate-") as temporary:
        directory = Path(temporary)
        junit = directory / "runtime.xml"
        result = subprocess.run(
            [
                str(python),
                "-m",
                "pytest",
                "tests/test_security_findings_toolset.py",
                "tests/test_findings_reader_toolset.py",
                "--junitxml",
                str(junit),
            ],
            cwd=root / "runtime",
            check=False,
        )
        if result.returncode != 0:
            raise GateError("Runtime findings tests failed")
        verify_python_report(junit)
        report = directory / "process.jsonl"
        pattern = "^(" + "|".join(PROCESS_TESTS) + ")$"
        with report.open("w") as output:
            process = subprocess.Popen(
                [
                    "go",
                    "test",
                    "-json",
                    "-tags=e2e",
                    "-count=1",
                    "-timeout=10m",
                    "./tests/e2e",
                    "-run",
                    pattern,
                ],
                cwd=root,
                stdout=subprocess.PIPE,
                text=True,
            )
            assert process.stdout is not None
            for line in process.stdout:
                output.write(line)
                output.flush()
                try:
                    event = json.loads(line)
                except ValueError:
                    print(line, end="", flush=True)
                    continue
                if event.get("Action") == "output":
                    print(event.get("Output", ""), end="", flush=True)
            status = process.wait()
        verify_go_report(report)
        if status != 0:
            raise GateError("Go findings processes failed")
    print(
        f"Findings gate passed: {len(REQUIRED_TESTS)} required process cases and Runtime contracts."
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--verify-go-report", type=Path, help="check a saved Go JSON stream offline"
    )
    parser.add_argument(
        "--verify-python-report", type=Path, help="check a saved pytest JUnit report"
    )
    args = parser.parse_args()
    try:
        if args.verify_go_report is not None:
            verify_go_report(args.verify_go_report)
        elif args.verify_python_report is not None:
            verify_python_report(args.verify_python_report)
        else:
            run(Path(__file__).resolve().parents[1])
    except (GateError, OSError) as error:
        print(f"Findings gate failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
