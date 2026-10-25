#!/usr/bin/env python3
"""Run complete sitemap owner packages and require every real-store case."""

import json
import os
import subprocess
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PREFIX = "github.com/grauwolf32/contractor/"
PACKAGES = (
    "internal/projectsitemap", "internal/artifacts", "internal/projectstore",
    "internal/projectlifecycle", "internal/persistence/postgres",
    "internal/httpapi/public", "internal/cli", "internal/publicclient",
    "internal/app", "internal/artifactpolicy", "internal/archtest",
)
REQUIRED = {
    "TestPostgresSitemapReceiptsOwnershipAndHistory",
    "TestPostgresSitemapTenWritersAndPermanentSourceConflict",
    "TestPostgresSitemapMalformedAndProjectFence",
    "TestPostgresSitemapSingleRunExceedsRemovedPublicationCaps",
    "TestPostgresSitemapSelectedContextUsesOrdinaryExactForkHold",
    "TestPostgresSitemapReplayKeepsOriginalStructuralPolicy",
    "TestPostgresSitemapUsesSharedTransferAdmission",
    "TestPostgresSitemapCommitsCommonMaintenanceAfterRollback",
}


def main():
    if not os.environ.get("CONTRACTOR_TEST_DATABASE_URL"):
        raise SystemExit("CONTRACTOR_TEST_DATABASE_URL is required")
    with tempfile.TemporaryDirectory(prefix="project-sitemap-postgres-") as directory:
        report = Path(directory) / "tests.jsonl"
        with report.open("w") as output:
            result = subprocess.run(
                ["go", "test", "-json", "-tags=integration", "-p=2", "-count=1",
                 "-timeout=15m", *("./" + package for package in PACKAGES)],
                cwd=ROOT, stdout=output, check=False, timeout=1200,
            )
        passed, started, packages, failures = set(), set(), set(), []
        domain = PREFIX + "internal/projectsitemap"
        for line in report.read_text().splitlines():
            event = json.loads(line)
            package, case, action = event.get("Package"), event.get("Test"), event.get("Action")
            if action == "pass" and not case:
                packages.add(package)
            if package == domain and case in REQUIRED:
                if action == "run":
                    started.add(case)
                if action == "pass":
                    passed.add(case)
                if action in {"skip", "fail"}:
                    failures.append(f"{case}: {action}")
            if action == "output" and ("encoded_publication_bytes=" in event.get("Output", "")):
                print(event["Output"].strip(), flush=True)
        missing = REQUIRED - (started & passed)
        missing_packages = {PREFIX + package for package in PACKAGES} - packages
        if result.returncode or missing or missing_packages or failures:
            # Connection strings are never diagnostics.
            print(report.read_text().replace(os.environ["CONTRACTOR_TEST_DATABASE_URL"], "[database]"))
            raise SystemExit(f"sitemap gate failed: {sorted(missing)}, {sorted(missing_packages)}, {failures}")
        print(f"Sitemap PostgreSQL: {len(PACKAGES)} complete packages, {len(REQUIRED)} mandatory cases, zero mandatory skips")


if __name__ == "__main__":
    main()
