#!/usr/bin/env python3
"""Require actual Server/Runtime/PostgreSQL sitemap process evidence."""

import os
import tempfile
from pathlib import Path

import release_test_reports as reports

ROOT = Path(__file__).resolve().parents[1]
PROCESS_TESTS = ("TestProjectSitemapToolsAcrossProductionProcesses", "TestProjectSitemapHTTPCollectionAcrossProductionProcesses", "TestProjectSitemapReportImportsAcrossProductionProcesses", "TestProjectSitemapServerOperationsAcrossProductionProcesses")
REQUIRED_TESTS = (*PROCESS_TESTS, *(
    PROCESS_TESTS[-1] + "/" + scenario
    for scenario in ("pure", "apply", "apply-only", "invalid")
))


def main():
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise SystemExit("CONTRACTOR_TEST_DATABASE_URL is required")
    with tempfile.TemporaryDirectory(prefix="sitemap-tools-e2e-") as directory:
        report = Path(directory) / "tests.jsonl"
        result = reports.capture(report, [
            "go", "test", "-json", "-tags=e2e", "-count=1", "-timeout=6m", "./tests/e2e",
            "-run", reports.pattern(PROCESS_TESTS),
        ], cwd=ROOT, timeout_seconds=420, max_report_bytes=32 << 20)
        if result:
            raise SystemExit(result)
        passed = reports.verify_go_report(report, {
            "github.com/grauwolf32/contractor/tests/e2e": list(REQUIRED_TESTS),
        })
        print(f"Sitemap processes: {len(passed)} passing cases, zero skips")


if __name__ == "__main__":
    main()
