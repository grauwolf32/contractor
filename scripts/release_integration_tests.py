#!/usr/bin/env python3
"""Discover integration-tagged Go tests selected by the release gate.

Only tests needing tools absent from the standard CI runner are excepted. Each
exception has an explicit, opt-in Make target; all other names enter one race
command so adding or renaming a tagged test cannot silently drop it.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = ("cmd", "internal", "tests", "tools")
TEST = re.compile(r"^func (Test\w+)\(", re.MULTILINE)
BUILD_TAG = re.compile(r"^//go:build (.+)$", re.MULTILINE)

# The target and reason are reviewed alongside the test instead of making a
# directory-wide exception, which would silently hide future regressions.
EXCEPTIONS = {
    ("internal/gitimport/real_git_integration_test.go", "TestRealGitReadOnlyContainer"): (
        "test-git-artifacts",
        "requires Podman and the pinned read-only container image",
    ),
    ("internal/gitimport/real_git_integration_test.go", "TestReadOnlyGitProbe"): (
        "test-git-artifacts",
        "runs only inside the read-only Podman container started by its parent",
    ),
    ("tests/integration/restore/restore_test.go", "TestBackupRestorePreservesExactArtifactsAndCAS"): (
        "test-backup-restore",
        "requires pg_dump, pg_restore and CREATE/DROP DATABASE privileges",
    ),
}


def discover() -> dict[str, set[str]]:
    selected: dict[str, set[str]] = {}
    found_exceptions: set[tuple[str, str]] = set()
    for source_root in SOURCE_ROOTS:
        for path in sorted((ROOT / source_root).rglob("*_test.go")):
            source = path.relative_to(ROOT).as_posix()
            data = path.read_text()
            tag = BUILD_TAG.search(data.split("\npackage ", 1)[0])
            if tag is None or not re.search(r"\bintegration\b", tag.group(1)):
                continue
            if tag.group(1) != "integration":
                raise ValueError(f"{source}: unsupported integration build constraint")
            names = TEST.findall(data)
            if not names:
                raise ValueError(f"{source}: integration file defines no Test functions")
            package = "./" + path.parent.relative_to(ROOT).as_posix()
            for name in names:
                key = (source, name)
                if key in EXCEPTIONS:
                    found_exceptions.add(key)
                else:
                    selected.setdefault(package, set()).add(name)
    if found_exceptions != EXCEPTIONS.keys():
        raise ValueError(f"stale integration exceptions: {sorted(EXCEPTIONS.keys() - found_exceptions)}")
    if not selected:
        raise ValueError("no release integration tests discovered")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--packages", action="store_true")
    parser.add_argument("--regex", action="store_true")
    args = parser.parse_args()
    if args.packages == args.regex:
        parser.error("choose --packages or --regex")
    selected = discover()
    if args.packages:
        print(" ".join(sorted(selected)))
    else:
        names = sorted({name for tests in selected.values() for name in tests})
        print("^(" + "|".join(names) + ")$")


if __name__ == "__main__":
    main()
