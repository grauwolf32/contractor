#!/usr/bin/env python3
"""List the Go packages of the second release race pass.

The first pass races the explicit RELEASE_RACE_PATTERNS in make/release.mk.
This pass races every other package with tests, so a new package enters the
race detector automatically. A package may stay out only through EXCEPTIONS,
which names the reason.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODULE = "github.com/grauwolf32/contractor/"
WITH_TESTS = "{{if or .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}"

# Packages whose tests never run under the race detector, with the reason.
EXCEPTIONS = {
    "tests/integration/restore": (
        "its only test is the opt-in backup/restore check, which needs pg_dump, "
        "pg_restore and CREATE DATABASE (make test-backup-restore)"
    ),
    "tests/ui-stack": (
        "its only tests are e2e-tagged browser process tests that drive separately "
        "built Server, Runtime and Node processes"
    ),
}


def go_list(*arguments: str) -> set[str]:
    result = subprocess.run(
        ["go", "list", "-e", *arguments], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return set(result.stdout.split())


def packages_with_tests(tags: str = "") -> set[str]:
    return go_list("-tags=" + tags, "-f", WITH_TESTS, "./...")


def discovered(excluded_patterns: list[str]) -> list[str]:
    excluded = go_list(*excluded_patterns) if excluded_patterns else set()
    exceptions = {MODULE + package for package in EXCEPTIONS}
    return sorted(packages_with_tests() - excluded - exceptions)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exclude", nargs="*", default=[], help="package patterns of the first race pass")
    print(" ".join(discovered(parser.parse_args().exclude)))


if __name__ == "__main__":
    main()
