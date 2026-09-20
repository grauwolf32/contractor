#!/usr/bin/env python3
"""Require every production-process OpenAPI Audit acceptance case to execute."""

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEST = "TestOpenAPIAuditScanAcrossProductionProcesses"
PACKAGE = "github.com/grauwolf32/contractor/tests/e2e"
CASES = (
    "sqlmap_exact_request_and_retained_package",
    "nuclei_url_coverage_and_pinned_template",
    "missing_concrete_input_never_dispatches",
    "lost_journal_ack_does_not_repeat_scan",
    "publication_and_collection_faults_survive_restart",
    "known_failure_has_bounded_cross_run_retries",
    "cancelled_scan_is_not_resubmitted",
)


def verify(path: Path) -> None:
    required = {TEST, *(f"{TEST}/{case}" for case in CASES)}
    started, passed = set(), set()
    package_passed = False
    for line in path.read_text().splitlines():
        event = json.loads(line)
        if event.get("Package") != PACKAGE:
            continue
        action, test = event.get("Action"), event.get("Test")
        if action in {"skip", "fail"}:
            raise ValueError(
                f"acceptance case did not pass: {test or PACKAGE} ({action})"
            )
        if action == "run" and test:
            started.add(test)
        if action == "pass":
            if test:
                passed.add(test)
            else:
                package_passed = True
    missing = required - (started & passed)
    if missing or not package_passed:
        raise ValueError(
            "acceptance cases absent: " + ", ".join(sorted(missing or {PACKAGE}))
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--verify-only", type=Path)
    args = parser.parse_args()
    if args.verify_only:
        verify(args.verify_only)
        return
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise ValueError("CONTRACTOR_TEST_DATABASE_URL is required")
    with tempfile.TemporaryDirectory(prefix="openapi-audit-gate-") as directory:
        report = args.report or Path(directory) / "process.jsonl"
        report.parent.mkdir(parents=True, exist_ok=True)
        with report.open("w") as output:
            result = subprocess.run(
                [
                    "go",
                    "test",
                    "-json",
                    "-tags=e2e",
                    "-count=1",
                    "-timeout=6m",
                    "./tests/e2e",
                    "-run",
                    f"^{TEST}$",
                ],
                cwd=ROOT,
                stdout=output,
                check=False,
            )
        if result.returncode:
            print(report.read_text())
            raise ValueError(f"process acceptance exited {result.returncode}")
        verify(report)
    print(f"OpenAPI Audit acceptance passed: {len(CASES)} mandatory process cases")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError) as error:
        raise SystemExit(f"OpenAPI Audit acceptance failed: {error}") from error
