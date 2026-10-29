#!/usr/bin/env python3
"""Require actual Server/Runtime/PostgreSQL sitemap process evidence."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROCESS_TESTS = ("TestProjectSitemapToolsAcrossProductionProcesses", "TestProjectSitemapHTTPCollectionAcrossProductionProcesses", "TestProjectSitemapReportImportsAcrossProductionProcesses")


def main():
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise SystemExit("CONTRACTOR_TEST_DATABASE_URL is required")
    with tempfile.TemporaryDirectory(prefix="sitemap-tools-e2e-") as directory:
        report = Path(directory) / "tests.jsonl"
        with report.open("w") as output:
            result = subprocess.run(
                ["go", "test", "-json", "-tags=e2e", "-count=1", "-timeout=6m", "./tests/e2e",
                 "-run", "^(" + "|".join(PROCESS_TESTS) + ")$"],
                cwd=ROOT, stdout=output, check=False, timeout=420,
            )
        started, passed, failures = set(), set(), []
        for line in report.read_text().splitlines():
            event = json.loads(line)
            case, action = event.get("Test"), event.get("Action")
            if case in PROCESS_TESTS:
                if action == "run":
                    started.add(case)
                if action == "pass":
                    passed.add(case)
                if action in {"skip", "fail"}:
                    failures.append(f"{case}: {action}")
        missing = set(PROCESS_TESTS) - (started & passed)
        if result.returncode or missing or failures:
            print(report.read_text().replace(os.environ["CONTRACTOR_TEST_DATABASE_URL"], "[database]"))
            raise SystemExit(f"sitemap process gate failed: {sorted(missing)}, {failures}")
        print(f"Sitemap processes: {len(passed)} mandatory scenarios, zero skips")


if __name__ == "__main__":
    main()
