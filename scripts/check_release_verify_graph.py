#!/usr/bin/env python3
"""Keep the release gate's deduplicated Go suites complete."""

import re
import shlex
import subprocess
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE = "github.com/grauwolf32/contractor/"

# Union of the untagged race packages selected by release-verify before V155.
EXPECTED_RACE_PACKAGES = {
    MODULE + relative
    for relative in (
        "cmd/contractor-skill",
        "internal/agentskills",
        "internal/app",
        "internal/artifactpolicy",
        "internal/artifacts",
        "internal/config",
        "internal/contracts",
        "internal/controlplane",
        "internal/credentials",
        "internal/credentials/litellm",
        "internal/httpapi",
        "internal/httpapi/artifacttransfer",
        "internal/httpapi/httpx",
        "internal/httpapi/privateartifacts",
        "internal/httpapi/public",
        "internal/httpapi/public/events",
        "internal/memory",
        "internal/mtls",
        "internal/performance",
        "internal/persistence/postgres",
        "internal/planner",
        "internal/planner/a2a",
        "internal/planner/router",
        "internal/planner/scan",
        "internal/planner/session",
        "internal/planner/stateview",
        "internal/planner/streamline",
        "internal/profiling",
        "internal/projectlifecycle",
        "internal/projectstore",
        "internal/runstore",
        "internal/runtimeconfig",
        "internal/scheduler",
        "internal/telemetry",
        "tests/integration/lease",
        "tools/performancebench",
    )
}

# Union of the process e2e names selected by release-verify before V155.
EXPECTED_PROCESS_TESTS = {
    "TestAgentSkillsMVPProcesses",
    "TestAuditProgramsAcrossProductionProcesses",
    "TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses",
    "TestHTTPAndCaidoAcrossHeterogeneousRuntimeProcesses",
    "TestHeterogeneousRuntimeCapabilityPlacement",
    "TestLabelDrivenRuntimeConfigurationAcrossProcesses",
    "TestLocalGoToPythonArtifactCopy",
    "TestProductionMemoryTemplatesAcrossProcesses",
    "TestProjectWorkspaceLifecycleAcrossProductionProcesses",
    "TestRoutingAndEscalationProductionBoundaries",
    "TestRunMetadataLabelsAcrossProcesses",
    "TestSchedulerConcurrencyAcrossProductionProcesses",
    "TestSharedMemoryMVPProcesses",
    "TestTaintAnnotationsAcrossRealRuntimeProcess",
    "TestWorkerSessionModesAcrossProductionProcesses",
    "TestWorkerSummarizerProductionBoundaries",
}


def dry_run(target: str) -> list[str]:
    result = subprocess.run(
        ["make", "-n", target], cwd=ROOT, capture_output=True, text=True, check=True
    )
    return result.stdout.splitlines()


def check_release_graph() -> None:
    commands = dry_run("release-verify")
    races: list[str] = []
    process_tests: Counter[str] = Counter()
    process_commands = 0
    for command in commands:
        if "go test" not in command:
            continue
        tokens = shlex.split(command)
        if "-race" in tokens and not any(token.startswith("-tags=") for token in tokens):
            races.append(command)
        if "-tags=e2e" in tokens and "./tests/e2e" in tokens:
            process_commands += 1
            pattern = tokens[tokens.index("-run") + 1]
            process_tests.update(re.findall(r"Test[A-Za-z0-9_]+", pattern))

    if len(races) != 1:
        raise SystemExit(f"release gate has {len(races)} untagged race commands, want one")
    patterns = re.search(r"\$\(go list (.*?) \| sort -u\)", races[0])
    if patterns is None or "-count=1" not in shlex.split(races[0]):
        raise SystemExit("release race command must deduplicate go list packages and disable cache")
    listed = subprocess.run(
        ["go", "list", *patterns.group(1).split()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    packages = set(listed.stdout.splitlines())
    if packages != EXPECTED_RACE_PACKAGES:
        raise SystemExit(
            "release race package set changed: "
            f"missing={sorted(EXPECTED_RACE_PACKAGES - packages)}, "
            f"added={sorted(packages - EXPECTED_RACE_PACKAGES)}"
        )
    if process_commands != 1 or set(process_tests) != EXPECTED_PROCESS_TESTS or any(
        count != 1 for count in process_tests.values()
    ):
        raise SystemExit(
            "release process test set changed: "
            f"commands={process_commands}, "
            f"missing={sorted(EXPECTED_PROCESS_TESTS - set(process_tests))}, "
            f"added={sorted(set(process_tests) - EXPECTED_PROCESS_TESTS)}, "
            f"duplicates={sorted(name for name, count in process_tests.items() if count != 1)}"
        )


def check_family_entry_points() -> None:
    for target, marker in (
        ("test-worker-session-modes-hardening", "go test -race -count=1"),
        ("test-project-workspaces-e2e", "go test -tags=e2e -count=1"),
        ("test-agent-skills-races", "go test -race -count=1"),
    ):
        if not any(marker in command for command in dry_run(target)):
            raise SystemExit(f"{target} lost its focused Go suite")


if __name__ == "__main__":
    check_release_graph()
    check_family_entry_points()
    print("release graph: 36 race packages and 16 process tests, each selected once")
