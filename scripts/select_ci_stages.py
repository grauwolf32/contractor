#!/usr/bin/env python3
"""Select PR checks from changed paths; tags and manual runs use the full gate."""

import json
import os
import subprocess
import sys
from pathlib import Path


# Keep this in make/release.mk stage order. The release-graph guard verifies it.
RELEASE_STAGE_TIMEOUTS = {
    "release-verify-lint": 30,
    "release-verify-unit": 45,
    "release-verify-ui": 30,
    "release-verify-families": 60,
    "release-verify-browser-a": 60,
    "release-verify-browser-b": 60,
    "release-verify-race": 45,
    "release-verify-race-discovered": 50,
    "release-verify-integration": 45,
    "release-verify-process-a": 80,
    "release-verify-process-b": 80,
}
PR_STAGE_TIMEOUTS = {"test-ui-stack-operations": 30}
STAGE_TIMEOUTS = {**RELEASE_STAGE_TIMEOUTS, **PR_STAGE_TIMEOUTS}

LINT = "release-verify-lint"
UNIT = "release-verify-unit"
UI = "release-verify-ui"
BROWSER_A = "release-verify-browser-a"
BROWSER_B = "release-verify-browser-b"
INTEGRATION = "release-verify-integration"
PROCESS_A = "release-verify-process-a"
PROCESS_B = "release-verify-process-b"
UI_STACK_OPERATIONS = "test-ui-stack-operations"


def stages_for_paths(paths: list[str]) -> list[str]:
    """Each PR gets lint plus checks for every touched product boundary."""
    selected = {LINT}
    for path in paths:
        if path.startswith("ui/"):
            selected.add(UI)
            if path.startswith("ui/e2e/"):
                if path == "ui/e2e/stack.spec.ts":
                    selected.add(UI_STACK_OPERATIONS)
                else:
                    selected.add(BROWSER_A)
                    if path == "ui/e2e/evals-stack.spec.ts":
                        selected.add(BROWSER_B)
            else:
                selected.update((BROWSER_A, BROWSER_B))
        elif path.startswith("tests/ui-stack/"):
            selected.update((BROWSER_A, BROWSER_B))
        elif path.startswith("tests/e2e/"):
            selected.update((UNIT, PROCESS_A, PROCESS_B))
        elif path.startswith(("runtime/", "cmd/", "internal/", "tests/integration/", "tests/eval/", "tools/")):
            selected.update((UNIT, INTEGRATION))
        elif path.startswith(("api/", "configs/")):
            selected.update((UNIT, UI, BROWSER_A, BROWSER_B, INTEGRATION))
        elif path in {"go.mod", "go.sum"}:
            selected.update((UNIT, INTEGRATION, PROCESS_A, PROCESS_B))
        elif path.startswith(("docs/", ".github/", "make/", "scripts/", "deploy/")) or path in {
            "Makefile",
            ".gitignore",
        }:
            continue
        else:
            # A new source tree must not silently skip its tests.
            selected.update((UNIT, UI, INTEGRATION))
    if BROWSER_A in selected:
        selected.discard(UI_STACK_OPERATIONS)
    return [stage for stage in STAGE_TIMEOUTS if stage in selected]


def matrix_for_event(event: str, paths: list[str]) -> dict[str, list[dict[str, str | int]]]:
    stages = stages_for_paths(paths) if event == "pull_request" else list(RELEASE_STAGE_TIMEOUTS)
    return {"include": [{"stage": stage, "timeout": STAGE_TIMEOUTS[stage]} for stage in stages]}


def changed_paths(base: str, head: str) -> list[str]:
    if not base or not head:
        raise ValueError("PR comparison requires base and head revisions")
    result = subprocess.run(
        ["git", "diff", "--name-only", "--no-renames", "-z", base, head],
        check=True,
        capture_output=True,
    )
    return [os.fsdecode(name) for name in result.stdout.split(b"\0") if name]


def main() -> None:
    event = os.environ["GITHUB_EVENT_NAME"]
    paths = changed_paths(os.environ.get("PR_BASE_SHA", ""), os.environ["GITHUB_SHA"]) if event == "pull_request" else []
    matrix = matrix_for_event(event, paths)
    output = Path(os.environ["GITHUB_OUTPUT"])
    with output.open("a") as stream:
        stream.write("matrix=" + json.dumps(matrix, separators=(",", ":")) + "\n")
    print(f"Selected CI stages for {event}: {', '.join(row['stage'] for row in matrix['include'])}", file=sys.stderr)


if __name__ == "__main__":
    main()
