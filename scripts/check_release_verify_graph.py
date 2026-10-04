#!/usr/bin/env python3
"""Keep the release gate's deduplicated Go suites complete."""

import importlib.util
import os
import re
import shlex
import subprocess
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from release_integration_tests import BUILD_TAG, EXCEPTIONS, TEST, discover
from release_race_packages import EXCEPTIONS as RACE_EXCEPTIONS
from release_race_packages import packages_with_tests


ROOT = Path(__file__).resolve().parents[1]
MODULE = "github.com/grauwolf32/contractor/"

# Union of the process e2e names selected by release-verify before V155.
EXPECTED_PROCESS_TESTS = {
    "TestAgentSkillsMVPProcesses",
    "TestAuditProgramCatalogReplacementRestartsServer",
    "TestAuditProgramsAcrossProductionProcesses",
    "TestCodeAnalysisAcrossHeterogeneousRuntimeProcesses",
    "TestGatewayRecoveryCancellationAndPermanentErrorAcrossProcesses",
    "TestGatewayRecoveryKeepsThreeQueuedRunsAcrossProcesses",
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

EXPECTED_CONFIG_TESTS = {
    "TestCodeAnalysisE2EConfigurationLoads",
    "TestDomainGatewayFindsNamedInputAfterParameterBlock",
    "TestDomainGatewayScriptedModelFailureAdvancesWithoutFixtureFailure",
    "TestProductionMemoryConfigurationStaging",
    "TestProjectWorkerBudgetMatchesPinnedPolicy",
    "TestRuntimeWorkRootEmptyAllowsPersistentOwnerLock",
}

# e2e-tagged tests that need tools absent from the CI runner. Each names the
# opt-in target that selects it and why release-verify cannot run it; every
# other e2e-tagged test must be selected by release-verify.
OPT_IN_E2E_TESTS = {
    "TestArtifactBlobBackendsContainers": (
        "test-artifact-blob-backends",
        "requires Podman to run the production containers",
    ),
    "TestGitArtifactsProductionContainers": (
        "test-git-artifacts",
        "requires Podman, native git and the pinned read-only container image",
    ),
    "TestKatanaDiscoveryAcrossProductionProcesses": (
        "test-scan-e2e",
        "requires the real katana scanner on PATH",
    ),
    "TestOpenAPIAuditScanAcrossProductionProcesses": (
        "test-openapi-audit-scan-e2e",
        "requires the real nuclei and sqlmap scanners on PATH",
    ),
    "TestPodmanSandboxAcrossProductionProcesses": (
        "test-podman-e2e",
        "requires rootless Podman and a preinstalled digest-pinned image",
    ),
    "TestScanToolsAcrossProductionProcesses": (
        "test-scan-e2e",
        "requires the real nuclei, naabu, sqlmap, ffuf and katana scanners on PATH",
    ),
}


CI_WORKFLOW = ROOT / ".github/workflows/ci.yml"
TESTING_GUIDE = ROOT / "docs/testing/README.md"
# make verify, split so CI reports lint, unit and UI failures separately.
FAST_STAGES = ("release-verify-lint", "release-verify-unit", "release-verify-ui")


def make(*arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    # Under make lint the guard is itself a recipe: drop the parent's flags
    # and level so a nested make neither inherits -k or -j nor prints
    # directory changes between commands.
    environment = {
        name: value
        for name, value in os.environ.items()
        if name not in {"MAKEFLAGS", "MFLAGS", "MAKELEVEL", "MAKEOVERRIDES"}
    }
    return subprocess.run(
        ["make", "--no-print-directory", *arguments],
        cwd=ROOT, env=environment, capture_output=True, text=True, check=check,
    )


def dry_run(*targets: str) -> list[str]:
    return make("-n", *targets).stdout.splitlines()


def make_prerequisites(target: str) -> list[str]:
    # -q exits 1 for an out-of-date goal; the printed database is complete.
    result = make("-pq", target, check=False)
    if result.returncode not in (0, 1):
        raise SystemExit(f"make -pq {target} failed: {result.stderr.strip()}")
    for line in result.stdout.splitlines():
        name, separator, rest = line.partition(":")
        if separator and name == target and "=" not in rest:
            return rest.split("|", 1)[0].split()
    raise SystemExit(f"make defines no rule for {target}")


def is_heavy(command: str) -> bool:
    """Race, integration, process, browser-stack and script-gate suites."""
    if "./tests/ui-stack" in command or "scripts/test-" in command:
        return True
    if "go test" not in command:
        return False
    tokens = shlex.split(command)
    if "-race" in tokens or any(token.startswith("-tags=") and "integration" in token for token in tokens):
        return True
    if "-tags=e2e" in tokens and "-run" in tokens:
        names = set(re.findall(r"Test[A-Za-z0-9_]+", tokens[tokens.index("-run") + 1]))
        return not names <= EXPECTED_CONFIG_TESTS
    return False


def check_stage_order() -> list[str]:
    stages = make_prerequisites("release-verify")
    if tuple(stages[: len(FAST_STAGES)]) != FAST_STAGES:
        raise SystemExit(f"release-verify must start with {FAST_STAGES}, got {stages}")
    fast = [target for stage in FAST_STAGES for target in make_prerequisites(stage)]
    if sorted(fast) != sorted(make_prerequisites("verify")):
        raise SystemExit(f"fast release stages run {fast}, not exactly make verify")
    fast_commands = dry_run(*FAST_STAGES)
    heavy = [command for command in fast_commands if is_heavy(command)]
    if heavy:
        raise SystemExit(f"fast release stages run heavy suites: {heavy}")
    if dry_run("release-verify")[: len(fast_commands)] != fast_commands:
        raise SystemExit("release-verify no longer runs lint, unit and UI checks before other stages")
    return stages


def top_level_block(text: str, key: str) -> str:
    match = re.search(rf"^{key}:\n((?:[ #].*\n|\n)*)", text, re.M)
    if match is None:
        raise SystemExit(f"CI workflow has no top-level {key}")
    return match.group(1)


def check_ci_workflow(stages: list[str], text: str) -> None:
    name = CI_WORKFLOW.relative_to(ROOT)
    # Only a pull request cancels its superseded run; every push to main keeps
    # its own concurrency group, so no merge loses its verdict.
    concurrency = top_level_block(text, "concurrency")
    group = re.search(r"^\s+group:\s*(.+)$", concurrency, re.M)
    cancel = re.search(r"^\s+cancel-in-progress:\s*(.+)$", concurrency, re.M)
    if (
        group is None
        or cancel is None
        or cancel.group(1).strip() != "${{ github.event_name == 'pull_request' }}"
        or "github.event_name == 'pull_request' && github.ref" not in group.group(1)
        or "github.run_id" not in group.group(1)
    ):
        raise SystemExit(f"{name} must cancel superseded runs only for pull requests")
    ci_stages = re.findall(r"^\s+- stage: (\S+)\s*$", text, re.M)
    if ci_stages != stages:
        raise SystemExit(f"{name} runs stages {ci_stages}, release-verify runs {stages}")
    for required, reason in (
        (r"^\s+fail-fast: false\s*$", "one failing stage must not cancel the others"),
        (r"^\s+make -k \$\{\{ matrix\.stage \}\}", "each job must run its stage with make -k"),
        (r"^\s+needs: stage\s*$", "the release-verify job must aggregate every stage"),
    ):
        if re.search(required, text, re.M) is None:
            raise SystemExit(f"{name}: {reason}")


# Scanners whose verdict depends on live advisory databases.
ADVISORY_SCANNERS = ("govulncheck", "pip-audit", "audit_runtime_dependencies.py")


def check_advisories_outside_release(release: list[str], advisories: list[str], workflow: str) -> None:
    """release-verify stays deterministic; the advisory job runs the live scans."""
    scans = [command for command in release if any(scanner in command for scanner in ADVISORY_SCANNERS)]
    if scans:
        raise SystemExit(f"release-verify runs live advisory scans; move them to make advisories: {scans}")
    for scanner in ("govulncheck", "audit_runtime_dependencies.py"):
        if not any(scanner in command for command in advisories):
            raise SystemExit(f"make advisories no longer runs {scanner}")
    name = CI_WORKFLOW.relative_to(ROOT)
    if re.search(r"^\s+(?:- )?run: make (?:-k )?advisories\s*$", workflow, re.M) is None:
        raise SystemExit(f"{name} must run make advisories in its own job")
    direct = [scanner for scanner in ADVISORY_SCANNERS if scanner in workflow]
    if direct:
        raise SystemExit(f"{name} runs advisory scanners outside make advisories: {direct}")
    floating = re.findall(r"^\s+runs-on:\s*(\S+-latest)\s*$", workflow, re.M)
    if floating:
        raise SystemExit(f"{name} must pin runner images, not {sorted(set(floating))}")


def check_documented_stages(stages: list[str], text: str) -> None:
    documented = re.findall(r"^\| `(release-verify-[a-z0-9-]+)` \|", text, re.M)
    if documented != stages:
        raise SystemExit(
            f"{TESTING_GUIDE.relative_to(ROOT)} documents stages {documented}, "
            f"release-verify runs {stages}"
        )


def check_release_graph() -> list[str]:
    commands = dry_run("release-verify")
    races: list[str] = []
    process_tests: Counter[str] = Counter()
    config_tests: Counter[str] = Counter()
    process_commands = 0
    config_commands = 0
    for command in commands:
        if "go test" not in command:
            continue
        tokens = shlex.split(command)
        if "-race" in tokens and not any(token.startswith("-tags=") for token in tokens):
            races.append(command)
        if "-tags=e2e" in tokens and "./tests/e2e" in tokens:
            pattern = tokens[tokens.index("-run") + 1]
            names = re.findall(r"Test[A-Za-z0-9_]+", pattern)
            if set(names).intersection(EXPECTED_CONFIG_TESTS):
                config_commands += 1
                config_tests.update(names)
            else:
                process_commands += 1
                process_tests.update(names)

    stacks = [command for command in commands if "go test" in command and "./tests/ui-stack" in command]
    if len(stacks) != 1:
        raise SystemExit(f"release gate runs the browser stack {len(stacks)} times, want once")
    if len(races) != 2:
        raise SystemExit(f"release gate has {len(races)} untagged race commands, want the explicit and discovered passes")
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
    if config_commands != 1 or set(config_tests) != EXPECTED_CONFIG_TESTS or any(
        count != 1 for count in config_tests.values()
    ):
        raise SystemExit(
            "release configuration test set changed: "
            f"commands={config_commands}, "
            f"missing={sorted(EXPECTED_CONFIG_TESTS - set(config_tests))}, "
            f"added={sorted(set(config_tests) - EXPECTED_CONFIG_TESTS)}"
        )
    return races


def substitution(command: str) -> str:
    """The first $(...) command substitution of a dry-run command."""
    start = command.index("$(")
    depth = 0
    for index in range(start + 1, len(command)):
        depth += {"(": 1, ")": -1}.get(command[index], 0)
        if depth == 0:
            return command[start + 2 : index]
    raise SystemExit(f"unbalanced command substitution: {command}")


def substituted_packages(command: str) -> set[str]:
    """Evaluate a race command's package list exactly as make's shell would."""
    result = subprocess.run(["sh", "-c", substitution(command)], cwd=ROOT, capture_output=True, text=True, check=True)
    return set(result.stdout.split())


def check_race_coverage(
    raced: list[set[str]],
    with_tests: set[str],
    tagged_only: set[str],
    integration_raced: set[str],
    exceptions: dict[str, str],
) -> int:
    """Every package with tests runs under -race in some release stage, once,
    unless it is an exception with a reason."""
    counts = Counter(package for packages in raced for package in packages)
    problems = [f"{package} is raced by {count} release passes" for package, count in sorted(counts.items()) if count > 1]
    excepted = {MODULE + package: reason for package, reason in exceptions.items()}
    for package in sorted(with_tests - counts.keys() - excepted.keys()):
        problems.append(f"{package} has tests but runs under -race in no release stage")
    for package in sorted(tagged_only - integration_raced - excepted.keys()):
        problems.append(f"{package} has only tagged tests and none run under -race")
    for package, reason in sorted(excepted.items()):
        if not reason:
            problems.append(f"race exception {package} has no reason")
        if package not in with_tests | tagged_only:
            problems.append(f"race exception {package} has no tests")
        if package in counts or package in integration_raced:
            problems.append(f"race exception {package} runs under -race; remove the exception")
    if problems:
        raise SystemExit(
            "release race coverage is incomplete (see scripts/release_race_packages.py):\n  "
            + "\n  ".join(problems)
        )
    return len(counts)


def race_command_packages(races: list[str]) -> list[set[str]]:
    raced = []
    for command in races:
        tokens = shlex.split(command)
        if "-count=1" not in tokens or "-run" in tokens:
            raise SystemExit(f"release race passes must race whole packages without the test cache: {command}")
        raced.append(substituted_packages(command))
    return raced


def check_family_entry_points() -> None:
    for target, marker in (
        ("test-worker-session-modes-hardening", "go test -race -count=1"),
        ("test-project-workspaces-e2e", "go test -tags=e2e -count=1"),
        ("test-agent-skills-races", "go test -race -count=1"),
        ("test-ui-stack", "./tests/ui-stack"),
    ):
        if not any(marker in command for command in dry_run(target)):
            raise SystemExit(f"{target} lost its focused Go suite")


@dataclass(frozen=True)
class GoTest:
    """One go test invocation: where it comes from, tags, packages, -run."""

    source: str
    tags: frozenset[str]
    packages: tuple[str, ...]
    run: str | None
    race: bool = False


def parse_go_test(command: str, source: str) -> GoTest | None:
    if "go test" not in command:
        return None
    tokens = shlex.split(command)
    start = next(index for index in range(len(tokens)) if tokens[index : index + 2] == ["go", "test"])
    arguments = tokens[start + 2 :]
    tags: frozenset[str] = frozenset()
    run = None
    for index, token in enumerate(arguments):
        if token.startswith("-tags="):
            tags = frozenset(filter(None, token.removeprefix("-tags=").split(",")))
        elif token == "-run":
            run = arguments[index + 1]
        elif token.startswith("-run="):
            run = token.removeprefix("-run=")
    packages = tuple(token for token in arguments if token.startswith("./"))
    return GoTest(source, tags, packages, run, "-race" in arguments)


def load_script(relative: str):
    spec = importlib.util.spec_from_file_location(Path(relative).stem.replace("-", "_"), ROOT / relative)
    if spec is None or spec.loader is None:
        raise SystemExit(f"cannot load {relative}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def script_go_tests(relative: str) -> list[GoTest]:
    """The Go selections a gate script runs, read from the script's own constants."""
    module = load_script(relative)
    if relative == "scripts/test-findings-e2e.py":
        return [GoTest(relative, frozenset({"e2e"}), ("./tests/e2e",), "^(" + "|".join(module.PROCESS_TESTS) + ")$")]
    if relative == "scripts/test-openapi-audit-scan-e2e.py":
        return [GoTest(relative, frozenset({"e2e"}), ("./tests/e2e",), f"^{module.TEST}$")]
    if relative == "scripts/test-audit-completion-e2e.py":
        packages = tuple("./" + name.removeprefix(MODULE) for name in module.matrix()["go"])
        return [GoTest(relative, frozenset({"integration"}), packages, module.PATTERN, race=True)]
    raise SystemExit(f"{relative} runs Go tests this guard cannot see; declare its selection in script_go_tests")


SCRIPT_GATE = re.compile(r"\bpython3 (scripts/test-[\w-]+\.py)\b")


def selections(*targets: str) -> list[GoTest]:
    found = []
    for command in dry_run(*targets):
        test = parse_go_test(command, " ".join(targets))
        if test is not None:
            found.append(test)
        for script in SCRIPT_GATE.findall(command):
            found.extend(script_go_tests(script))
    return found


def split_top(pattern: str, separator: str) -> list[str]:
    """Split a regex at separators outside groups, classes and escapes."""
    parts, current, depth, in_class, escaped = [], "", 0, False, False
    for char in pattern:
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
        elif char == separator and depth == 0:
            parts.append(current)
            current = ""
            continue
        current += char
    return [*parts, current]


def wraps_whole(pattern: str) -> bool:
    """Whether pattern is ^( ... )$ with one group spanning everything between."""
    if not (pattern.startswith("^(") and pattern.endswith(")$")):
        return False
    depth, in_class, escaped = 0, False, False
    for index in range(1, len(pattern) - 1):
        char = pattern[index]
        if escaped:
            escaped = False
        elif char == "\\":
            escaped = True
        elif in_class:
            in_class = char != "]"
        elif char == "[":
            in_class = True
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0 and index != len(pattern) - 2:
                return False
    return depth == 0


def run_alternatives(pattern: str) -> list[str]:
    """Top-level test alternatives of a -run pattern; go test matches each by search."""
    level = split_top(pattern, "/")[0]
    if wraps_whole(level):
        return [f"^(?:{alternative})$" for alternative in split_top(level[2:-2], "|")]
    return split_top(level, "|")


def selects(test: GoTest, name: str) -> bool:
    return test.run is None or re.search(split_top(test.run, "/")[0], name) is not None


class Inventory:
    """go list and go test -list results, cached per build-tag set."""

    def __init__(self) -> None:
        self.expanded: dict[tuple[frozenset[str], tuple[str, ...]], tuple[str, ...]] = {}
        self.listed: dict[tuple[frozenset[str], str], frozenset[str]] = {}

    def packages(self, tags: frozenset[str], patterns: tuple[str, ...]) -> tuple[str, ...]:
        key = (tags, patterns)
        if key not in self.expanded:
            # Packages whose files all need other tags have nothing to list.
            result = subprocess.run(
                [
                    "go", "list", "-e", "-tags=" + ",".join(sorted(tags)),
                    "-f", "{{if or .GoFiles .TestGoFiles .XTestGoFiles}}{{.ImportPath}}{{end}}",
                    *patterns,
                ],
                cwd=ROOT, capture_output=True, text=True, check=True,
            )
            self.expanded[key] = tuple(result.stdout.split())
        return self.expanded[key]

    def tests(self, tags: frozenset[str], packages: tuple[str, ...]) -> dict[str, frozenset[str]]:
        missing = sorted({package for package in packages if (tags, package) not in self.listed})
        if missing:
            result = subprocess.run(
                ["go", "test", "-list", ".", "-tags=" + ",".join(sorted(tags)), *missing],
                cwd=ROOT, capture_output=True, text=True,
            )
            if result.returncode:
                raise SystemExit(f"go test -list failed:\n{result.stdout}{result.stderr}")
            names: list[str] = []
            for line in result.stdout.splitlines():
                fields = line.split()
                if fields and fields[0] in {"ok", "?"}:
                    self.listed[(tags, fields[1])] = frozenset(names)
                    names = []
                elif fields:
                    names.append(fields[0])
        return {package: self.listed[(tags, package)] for package in packages}


def check_selected_tests_exist(tests: list[GoTest], inventory: Inventory) -> int:
    """Every -run alternative must select an existing test in its packages."""
    absent = []
    checked = 0
    # One go test -list per tag set instead of one per selection.
    wanted: dict[frozenset[str], set[str]] = {}
    for test in tests:
        if test.run is not None:
            wanted.setdefault(test.tags, set()).update(inventory.packages(test.tags, test.packages))
    for tags, packages in wanted.items():
        inventory.tests(tags, tuple(packages))
    for test in tests:
        if test.run is None:
            continue
        packages = inventory.packages(test.tags, test.packages)
        names = frozenset().union(*inventory.tests(test.tags, packages).values())
        for alternative in run_alternatives(test.run):
            # '^$' or '.*' select nothing or everything rather than a name.
            if re.fullmatch(alternative, ""):
                continue
            checked += 1
            if not any(re.search(alternative, name) for name in names):
                absent.append(f"{test.source}: {alternative!r} in {' '.join(test.packages)}")
    if absent:
        raise SystemExit("go test -run selections name no existing test:\n  " + "\n  ".join(absent))
    return checked


def e2e_tagged_tests(inventory: Inventory) -> dict[str, frozenset[str]]:
    """Tests compiled only with -tags=e2e, per package import path."""
    patterns = sorted(
        {
            "./" + path.parent.relative_to(ROOT).as_posix()
            for source_root in GO_SOURCE_ROOTS
            for path in (ROOT / source_root).rglob("*_test.go")
            if (tag := BUILD_TAG.search(path.read_text().split("\npackage ", 1)[0]))
            and re.search(r"\be2e\b", tag.group(1))
        }
    )
    e2e = frozenset({"e2e"})
    packages = inventory.packages(e2e, tuple(patterns))
    tagged = inventory.tests(e2e, packages)
    plain = inventory.tests(frozenset(), inventory.packages(frozenset(), tuple(patterns)))
    return {package: names - plain.get(package, frozenset()) for package, names in tagged.items()}


def check_e2e_reachable(
    tagged: dict[str, frozenset[str]],
    release: list[GoTest],
    opt_in: dict[str, list[GoTest]],
    inventory: Inventory,
) -> int:
    """Every e2e-tagged test runs in release-verify or in its allowlisted opt-in gate."""

    def selected(tests: list[GoTest], package: str, name: str) -> bool:
        return any(
            "e2e" in test.tags and package in inventory.packages(test.tags, test.packages) and selects(test, name)
            for test in tests
        )

    problems = []
    known = {name for names in tagged.values() for name in names}
    for name in sorted(OPT_IN_E2E_TESTS.keys() - known):
        problems.append(f"{name} is allowlisted but no longer exists")
    for package, names in sorted(tagged.items()):
        for name in sorted(names):
            reached = selected(release, package, name)
            if name not in OPT_IN_E2E_TESTS:
                if not reached:
                    problems.append(f"{name} ({package}) is not selected by release-verify")
                continue
            target, reason = OPT_IN_E2E_TESTS[name]
            if not reason:
                problems.append(f"{name} is allowlisted without a reason")
            if reached:
                problems.append(f"{name} runs in release-verify; remove it from the opt-in allowlist")
            if not selected(opt_in.get(target, []), package, name):
                problems.append(f"{name} is not selected by its opt-in target {target}")
    if problems:
        raise SystemExit("e2e-tagged test gates are incomplete:\n  " + "\n  ".join(problems))
    return len(known)


def race_constrained_packages() -> dict[str, frozenset[str]]:
    """Packages with test files built only with or without -race, and the
    other build tags their tests need."""
    constrained: dict[str, frozenset[str]] = {}
    tags: dict[str, set[str]] = {}
    for source_root in GO_SOURCE_ROOTS:
        for path in (ROOT / source_root).rglob("*_test.go"):
            tag = BUILD_TAG.search(path.read_text().split("\npackage ", 1)[0])
            if tag is None:
                continue
            package = "./" + path.parent.relative_to(ROOT).as_posix()
            names = re.findall(r"(!?)\b([A-Za-z_][A-Za-z0-9_.]*)\b", tag.group(1))
            tags.setdefault(package, set()).update(name for negated, name in names if not negated and name != "race")
            if any(name == "race" for _, name in names):
                constrained[package] = frozenset()
    return {package: frozenset(tags[package]) for package in constrained}


def check_non_race_passes(
    constrained: dict[str, frozenset[str]], release: list[GoTest], inventory: Inventory
) -> None:
    """A test budget relaxed under the race detector must also run without it."""
    missing = [
        package
        for package, tags in sorted(constrained.items())
        if not any(
            not test.race
            and test.run is None
            and tags <= test.tags
            and set(inventory.packages(test.tags, (package,))) <= set(inventory.packages(test.tags, test.packages))
            for test in release
        )
    ]
    if missing:
        raise SystemExit(
            "packages with race-constrained tests have no complete release pass without -race "
            f"(tags {[sorted(constrained[package]) for package in missing]}): {missing}"
        )


GO_SOURCE_ROOTS = ("cmd", "internal", "tests", "tools")
# A configured but unreachable database must fail the gate; a database test
# may skip only when CONTRACTOR_TEST_DATABASE_URL is unset.
DATABASE_CONNECT = re.compile(r"\.Ping\(|pgxpool\.New|pgx\.Connect|\.Acquire\(|sql\.Open\(")
ERROR_BRANCH = re.compile(r"^\s*if\b.*\berr\s*!=\s*nil\s*\{\s*$")
SKIP_CALL = re.compile(r"\.Skip(?:f|Now)?\(")
UNREACHABLE_SKIP = re.compile(
    r"\.Skip(?:f)?\(\s*\"[^\"]*(?:PostgreSQL|Postgres|database)[^\"]*(?:unavailable|unreachable|down|refused)",
    re.IGNORECASE,
)


def database_skip_violations(source: str) -> list[int]:
    """Return the lines of skips taken when a configured database is unreachable."""
    lines = source.splitlines()
    violations = {number for number, line in enumerate(lines, 1) if UNREACHABLE_SKIP.search(line)}
    for index, line in enumerate(lines):
        previous = lines[index - 1] if index else ""
        if ERROR_BRANCH.match(line) is None or not (
            DATABASE_CONNECT.search(line) or DATABASE_CONNECT.search(previous)
        ):
            continue
        depth = 0
        for offset in range(index, len(lines)):
            if offset > index and SKIP_CALL.search(lines[offset]):
                violations.add(offset + 1)
            depth += lines[offset].count("{") - lines[offset].count("}")
            if depth <= 0:
                break
    return sorted(violations)


def check_database_tests_fail_closed() -> None:
    found = {
        str(path.relative_to(ROOT)): lines
        for source_root in GO_SOURCE_ROOTS
        for path in sorted((ROOT / source_root).rglob("*_test.go"))
        if (lines := database_skip_violations(path.read_text()))
    }
    if found:
        raise SystemExit(
            "database tests skip when CONTRACTOR_TEST_DATABASE_URL is set but unreachable; "
            f"fail instead: {found}"
        )


def check_integration_graph() -> int:
    selected = discover()
    # The non-race budget pass is checked by check_non_race_passes.
    commands = [
        shlex.split(command)
        for command in dry_run("release-verify")
        if "go test" in command and "-tags=integration" in command and "-race" in shlex.split(command)
    ]
    consolidated = [tokens for tokens in commands if "-v" in tokens and "-p" in tokens]
    if len(commands) != 4 or len(consolidated) != 1:
        raise SystemExit(
            "release integration graph changed: "
            f"{len(commands)} visible commands, {len(consolidated)} complete discovery gates"
        )
    tokens = consolidated[0]
    if any(flag not in tokens for flag in ("-race", "-v", "-count=1", "-p", "1", "-run")):
        raise SystemExit("release integration command lost race, verbose evidence, count, serial or selection")
    packages = {token for token in tokens if token.startswith("./")}
    if packages != selected.keys():
        raise SystemExit(
            "release integration packages changed: "
            f"missing={sorted(selected.keys() - packages)}, "
            f"added={sorted(packages - selected.keys())}"
        )
    pattern = tokens[tokens.index("-run") + 1]
    regex = re.compile(pattern)
    expected = {name for names in selected.values() for name in names}
    named = set(re.findall(r"Test[A-Za-z0-9_]+", pattern))
    if named != expected or any(regex.fullmatch(name) is None for name in expected):
        raise SystemExit(
            "release integration test selection changed: "
            f"missing={sorted(expected - named)}, added={sorted(named - expected)}"
        )
    # A same-named untagged test in another selected package would also run in
    # the broad regex, repeating work already covered by test-release-go-race.
    for package in selected:
        for source in (ROOT / package.removeprefix("./")).glob("*_test.go"):
            data = source.read_text()
            if BUILD_TAG.search(data.split("\npackage ", 1)[0]) is None:
                repeated = expected.intersection(TEST.findall(data))
                if repeated:
                    raise SystemExit(f"integration regex also selects untagged tests in {source}: {sorted(repeated)}")
    for (source, name), (target, reason) in EXCEPTIONS.items():
        if not reason:
            raise SystemExit(f"integration exception {name} has no reason")
        package = "./" + str(Path(source).parent)
        if not any(
            "go test" in command
            and "-tags=integration" in command
            and package in shlex.split(command)
            and (
                "-run" not in shlex.split(command)
                or re.compile(shlex.split(command)[shlex.split(command).index("-run") + 1]).fullmatch(name)
            )
            for command in dry_run(target)
        ):
            raise SystemExit(f"integration exception {name} lost its {target} gate")
    return len(expected)


if __name__ == "__main__":
    stages = check_stage_order()
    check_ci_workflow(stages, CI_WORKFLOW.read_text())
    check_documented_stages(stages, TESTING_GUIDE.read_text())
    check_advisories_outside_release(dry_run("release-verify"), dry_run("advisories"), CI_WORKFLOW.read_text())
    races = check_release_graph()
    with_tests = packages_with_tests()
    tagged_only = (packages_with_tests("integration") | packages_with_tests("e2e")) - with_tests
    integration_raced = {MODULE + package.removeprefix("./") for package in discover()}
    raced = check_race_coverage(race_command_packages(races), with_tests, tagged_only, integration_raced, RACE_EXCEPTIONS)
    check_family_entry_points()
    inventory = Inventory()
    release = selections("release-verify")
    opt_in = {target: selections(target) for target in sorted({target for target, _ in OPT_IN_E2E_TESTS.values()})}
    named = check_selected_tests_exist(release + [test for tests in opt_in.values() for test in tests], inventory)
    tagged = check_e2e_reachable(e2e_tagged_tests(inventory), release, opt_in, inventory)
    check_non_race_passes(race_constrained_packages(), release, inventory)
    count = check_integration_graph()
    check_database_tests_fail_closed()
    print(
        f"release graph: {len(stages)} stages in CI order, {raced} race packages, 19 process tests, "
        f"6 fixture tests, {named} existing -run selections, {tagged} tagged e2e tests "
        f"({len(OPT_IN_E2E_TESTS)} opt-in) and {count} integration tests covered"
    )
